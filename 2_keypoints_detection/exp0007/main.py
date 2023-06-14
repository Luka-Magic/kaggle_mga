# basic
import sys
import gc
import numpy as np
from pathlib import Path
import json
import os
import zipfile
import random
from tqdm import tqdm
from collections import OrderedDict, Counter
import lmdb
import six
from PIL import Image
from torchsummary import summary
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Tuple, Any

# hydra
import hydra
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize_config_dir

# wandb
import wandb

# sklearn
from sklearn.model_selection import KFold, StratifiedKFold

# pytorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch import optim
from torchvision.transforms import Compose
from torch.cuda.amp import autocast, GradScaler

# other
import albumentations
from albumentations import KeypointParams
from albumentations.pytorch import ToTensorV2

from utils import seed_everything, AverageMeter, calc_accuracy, is_nan, tensor2arr, PointCounter
from pose_resnet import get_pose_net
from pspnet import PSPNet
from loss import CenterLoss, CenterSourceWeightLoss


thresholds = np.round(np.arange(0.1, 1.0, 0.1), 4).tolist()
wandb_thr = 0.3

CHART_TYPE2LABEL = {
    'line': 0,
    'vertical_bar': 1,
    'scatter': 2,
    'dot': 3,
    'horizontal_bar': 4
}


def split_data(cfg, lmdb_dir) -> Dict[int, Dict[str, Any]]:
    """
    データからextractedだけを抜き取りkfoldでsplitさせる
    その後trainデータにgeneratedのデータを全て合わせる

    Returns:
        indices_dict (Dict[int, Dict[str, List[int]]]): A dictionary containing train/valid indices of each fold.
            Example: {1: {'train': [0, 1, 4, ...], 'valid': [5, 7, ....]}
    """
    indices_dict = {}

    env = lmdb.open(str(lmdb_dir), max_readers=32,
                    readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))

    generated_indicies = []
    extracted_indices = []

    stratified_label = []
    if cfg.debug:
        n_samples = 5000

    for idx in tqdm(range(n_samples), total=n_samples):
        with env.begin(write=False) as txn:
            # load json
            label_key = f'label-{str(idx+1).zfill(8)}'.encode()
            label = txn.get(label_key).decode('utf-8')
        json_dict = json.loads(label)

        if json_dict['chart-type'] not in cfg.chart_types:
            continue

        label_source = json_dict['source']

        if label_source == 'extracted':
            extracted_indices.append(idx)
            stratified_label.append(CHART_TYPE2LABEL[json_dict['chart-type']])
        elif label_source == 'generated':
            generated_indicies.append(idx)

    if cfg.split_method == 'StratifiedKFold':
        for fold, (train_fold_indices, valid_fold_indices) \
                in enumerate(StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed).split(extracted_indices, stratified_label)):
            if cfg.debug:
                train_indices = [extracted_indices[i]
                                 for i in train_fold_indices]
                valid_indices = [extracted_indices[i]
                                 for i in valid_fold_indices]
            else:
                train_indices = [extracted_indices[i]
                                 for i in train_fold_indices] + generated_indicies
                valid_indices = [extracted_indices[i]
                                 for i in valid_fold_indices]
            indices_dict[fold] = {
                'train': train_indices,
                'valid': valid_indices,
            }
    else:
        NotImplementedError

    return indices_dict


def split_extra_data(cfg, extra_train_dirs, extra_valid_dirs):
    """
    Returns:
        extra_train_info (Dict[int, Dict[str, List[int]]]): A dictionary containing train indices of each fold.
            Example: {dataset_name: {'lmdb_dir': Path, 'train': [0, 1, 4, ...]}, ...}
        extra_valid_info (Dict[int, Dict[str, List[int]]]): A dictionary containing valid indices of each fold.
            Example: {dataset_name: {'lmdb_dir': Path, 'valid': [5, 7, ....]}, ...}
    """

    extra_train_info = {}  # key: dataset, value: {'train': []}
    extra_valid_info = {}  # key: dataset, value: {'valid': []}

    # train
    for extra_train_dir in extra_train_dirs:
        dataset_name = extra_train_dir.stem
        extra_train_info[dataset_name] = {'train': []}
        if (extra_train_dir / 'lmdb').exists():
            lmdb_dir = extra_train_dir / 'lmdb'
        else:
            lmdb_dir = extra_train_dir / 'train' / 'lmdb'

        extra_train_info[dataset_name]['lmdb_dir'] = lmdb_dir
        env = lmdb.open(str(lmdb_dir), max_readers=32,
                        readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            n_samples = int(txn.get('num-samples'.encode()))

        for idx in tqdm(range(n_samples), total=n_samples):
            with env.begin(write=False) as txn:
                # load json
                label_key = f'label-{str(idx+1).zfill(8)}'.encode()
                label = txn.get(label_key).decode('utf-8')
            json_dict = json.loads(label)

            if json_dict['chart-type'] not in cfg.chart_types:
                continue

            extra_train_info[dataset_name]['train'].append(idx)

    # valid
    for extra_valid_dir in extra_valid_dirs:
        dataset_name = extra_valid_dir.stem
        extra_valid_info[dataset_name] = {'valid': []}
        if (extra_valid_dir / 'lmdb').exists():
            lmdb_dir = extra_valid_dir / 'lmdb'
        else:
            lmdb_dir = extra_valid_dir / 'valid' / 'lmdb'

        extra_valid_info[dataset_name]['lmdb_dir'] = lmdb_dir

        env = lmdb.open(str(lmdb_dir), max_readers=32,
                        readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            n_samples = int(txn.get('num-samples'.encode()))

        for idx in tqdm(range(n_samples), total=n_samples):
            with env.begin(write=False) as txn:
                # load json
                label_key = f'label-{str(idx+1).zfill(8)}'.encode()
                label = txn.get(label_key).decode('utf-8')
            json_dict = json.loads(label)

            if json_dict['chart-type'] not in cfg.chart_types:
                continue

            extra_valid_info[dataset_name]['valid'].append(idx)
    return extra_train_info, extra_valid_info


# Lmdb Dataset
class MgaLmdbDataset(Dataset):
    def __init__(self, cfg, lmdb_dir, indices, transforms, phase='train'):
        super().__init__()
        self.cfg = cfg
        self.transforms = transforms
        self.indices = indices
        self.env = lmdb.open(str(lmdb_dir), max_readers=32,
                             readonly=True, lock=False, readahead=False, meminit=False)
        self.output_size = cfg.output_size
        self.sigma = cfg.sigma
        self.img_h, self.img_w = cfg.img_h, cfg.img_w
        self.heatmap_h, self.heatmap_w = cfg.heatmap_h, cfg.heatmap_w
        self.phase = phase
        self.chart2point_name = {
            'scatter': 'scatter points',
            'line': 'lines',
            'dot': 'dot points',
            'vertical_bar': 'bars',
            'horizontal_bar': 'bars',
        }

    def _overlap_heatmap(self, heatmap, center, sigma):
        tmp_size = sigma * 6
        mu_x = int(center[0] + 0.5)
        mu_y = int(center[1] + 0.5)
        w, h = heatmap.shape[0], heatmap.shape[1]
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
            return heatmap
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
        img_x = max(0, ul[0]), min(br[0], h)
        img_y = max(0, ul[1]), min(br[1], w)
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
        return heatmap

    def _create_heatmap(self, joints):
        '''
            joints: [(x1, y1), (x2, y2), ...]
            heatmap: size: (hm_h, hm_w)
        '''
        heatmap = np.zeros((self.heatmap_h, self.heatmap_w), dtype=np.float32)
        for joint_id in range(len(joints)):
            heatmap = self._overlap_heatmap(
                heatmap, joints[joint_id], self.sigma)
        return heatmap

    def _count_n_points(self, json_dict):
        """
        Args:
            json_dict (Dict[str, Any]): ターゲットのdict
        Returns:
            gt_string (str): 入力となるプロンプト
        """
        n_points = 0

        for d in json_dict['data-series']:
            x = d["x"]
            y = d["y"]
            # Ignore nan values
            if is_nan(x) or is_nan(y):
                continue
            n_points += 1

        return n_points

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        with self.env.begin(write=False) as txn:
            # load image
            img_key = f'image-{str(idx+1).zfill(8)}'.encode()
            imgbuf = txn.get(img_key)

            # load json
            label_key = f'label-{str(idx+1).zfill(8)}'.encode()
            label = txn.get(label_key).decode('utf-8')

        # image
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        if self.cfg.input_size == 3:
            img = np.array(Image.open(buf).convert('RGB'))
        else:
            img = np.array(Image.open(buf).convert('L'))
        h, w, _ = img.shape

        # label
        json_dict = json.loads(label)
        n_points = self._count_n_points(json_dict)
        source = 1 if json_dict['source'] == 'extracted' else 0

        if self.phase == 'train':
            chart_type = json_dict['chart-type']
            point_name = self.chart2point_name[chart_type]

            keypoints = []
            for dic in json_dict['visual-elements'][point_name][0]:
                x, y = dic['x'], dic['y']
                if x < 0 or y < 0 or x > w or y > h:
                    continue
                keypoints.append([x, y])

            transformed = self.transforms(image=img, keypoints=keypoints)
            img = transformed['image']
            img = torch.from_numpy(img).permute(2, 0, 1)
            keypoints = transformed['keypoints']

            if len(keypoints) != 0:
                keypoints_on_hm = np.array(keypoints) * \
                    np.array([self.heatmap_w, self.heatmap_h]) / \
                    np.array([self.img_w, self.img_h])

                heatmap = self._create_heatmap(keypoints_on_hm)
            else:
                heatmap = np.zeros((self.heatmap_h, self.heatmap_w))

            heatmap = torch.from_numpy(heatmap)

            return img, heatmap, n_points, source
        else:
            transformed = self.transforms(image=img)
            img = transformed['image']
            img = torch.from_numpy(img).permute(2, 0, 1)
            return img, n_points


def get_transforms(cfg, phase):
    if phase == 'train':
        aug = cfg.train_aug
    elif phase == 'valid':
        aug = cfg.valid_aug
    elif phase == 'tta':
        aug = cfg.tta_aug

    augs = [getattr(albumentations, name)(**kwargs)
            for name, kwargs in aug.items()]
    # augs.append(ToTensorV2(p=1.))
    if phase == 'train':
        return albumentations.Compose(augs, keypoint_params=KeypointParams(format='xy'))
    else:
        return albumentations.Compose(augs)


def prepare_dataloader(cfg, lmdb_dir, train_indices, valid_indices, extra_train_info, extra_valid_info):
    train_ds_list = []
    train_ds_list.append(MgaLmdbDataset(cfg, lmdb_dir, train_indices,
                                        transforms=get_transforms(cfg, 'train'), phase='train'))
    for dataset_info in extra_train_info.values():
        extra_lmdb_dir = dataset_info['lmdb_dir']
        extra_train_indices = dataset_info['train']
        train_ds_list.append(MgaLmdbDataset(cfg, extra_lmdb_dir, extra_train_indices,
                                            transforms=get_transforms(cfg, 'train'), phase='train'))
    concat_train_ds = ConcatDataset(train_ds_list)
    train_loader = DataLoader(
        concat_train_ds,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    valid_ds = MgaLmdbDataset(cfg, lmdb_dir, valid_indices,
                              transforms=get_transforms(cfg, 'valid'), phase='valid')
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.valid_bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    extra_valid_dict = {}
    for dataset_name, dataset_info in extra_valid_info.items():
        extra_lmdb_dir = dataset_info['lmdb_dir']
        extra_valid_indices = dataset_info['valid']
        extra_valid_ds = MgaLmdbDataset(
            cfg, extra_lmdb_dir, extra_valid_indices, transforms=get_transforms(cfg, 'valid'), phase='valid')
        extra_valid_loader = DataLoader(
            extra_valid_ds,
            batch_size=cfg.valid_bs,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        extra_valid_dict[dataset_name] = {
            'valid_loader':  extra_valid_loader
        }

    return train_loader, valid_loader, extra_valid_dict


def train_one_epoch(cfg, epoch, dataloader, model, loss_fn, device, optimizer, scheduler, scheduler_step_time, scaler, point_counter):
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    model.train()

    # accuracy = AverageMeter()
    losses = AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, (images, heatmaps, n_points, source) in pbar:
        images = images.to(device).float()
        heatmaps = heatmaps.to(device).float()
        source = source.to(device).long()
        bs = len(images)
        with autocast(enabled=cfg.use_amp):
            pred = model(images)
            loss = loss_fn(pred, heatmaps, source)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # avg_acc, cnt = calc_accuracy(
        #     pred.detach().cpu().numpy(), heatmaps.detach().cpu().numpy())

        # pred_n_points, _ = point_counter.count(pred, 3.)
        # gt_n_points = n_points.numpy()

        # acc = np.mean(pred_n_points == gt_n_points)

        # accuracy.update(acc, bs)
        losses.update(loss.item(), bs)
        lr = get_lr(optimizer)
        if scheduler_step_time == 'step':
            scheduler.step()
        pbar.set_description(f'[Train epoch {epoch}/{cfg.n_epochs}]')
        # pbar.set_postfix(OrderedDict(loss=losses.avg, acc=accuracy.avg))
        if cfg.use_wandb:
            wandb.log({
                'step': (epoch - 1) * len(pbar) + step,
                # 'train_accuracy': accuracy.avg,
                'train_loss': losses.avg,
                'lr': lr
            })
    if scheduler_step_time == 'epoch':
        scheduler.step()

    lr = get_lr(optimizer)

    return losses.avg, lr


def valid_one_epoch(cfg, epoch, dataloader, model, loss_fn, device, point_counter):
    model.eval()

    acc_per_thr = {thr: AverageMeter() for thr in thresholds}

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    idx = 1
    for step, (images, n_points) in pbar:
        images = images.to(device).float()
        bs = len(images)

        with torch.no_grad():
            pred = model(images)

        gt_n_points = n_points.numpy()

        # wandb_score_maps = []
        # wandb_n_preds = []
        pred_n_points, score_map = point_counter.count(pred, thresholds)
        for thr in thresholds:
            acc = np.mean(pred_n_points[thr] == gt_n_points)
            acc_per_thr[thr].update(acc, bs)

        wandb_n_pred = pred_n_points[wandb_thr]

        pbar.set_description(f'[Valid epoch {epoch}/{cfg.n_epochs}]')
        pbar.set_postfix(OrderedDict(accuracy=acc_per_thr[wandb_thr].avg))

        if cfg.use_wandb:
            for i in range(bs):
                wandb.log({
                    'image': wandb.Image(tensor2arr(images[i].detach().cpu(), cfg.img_mean, cfg.img_std)),
                    'pred_heatmap': wandb.Image(
                        torch.sigmoid(pred[i]).detach().cpu().numpy(),
                        caption=f'gt: {gt_n_points[i]} / pred: {wandb_n_pred[i]} (thr={wandb_thr})'
                    ),
                    'score_map': wandb.Image(score_map[i])
                })

    return acc_per_thr


def main():
    EXP_PATH = Path.cwd()
    with initialize_config_dir(config_dir=str(EXP_PATH / 'config')):
        cfg = compose(config_name='config.yaml')

    ROOT_DIR = Path.cwd().parents[2]
    exp_name = EXP_PATH.name
    LMDB_DIR = ROOT_DIR / 'data' / cfg.dataset_name / 'lmdb'
    EXTRA_TRAIN_DIRS = []
    EXTRA_VALID_DIRS = []
    for extra_train_dir in cfg.extra_train_datasets:
        EXTRA_TRAIN_DIRS.append(ROOT_DIR / 'data' / extra_train_dir)
    for extra_valid_dir in cfg.extra_valid_datasets:
        EXTRA_VALID_DIRS.append(ROOT_DIR / 'data' / extra_valid_dir)
    SAVE_DIR = ROOT_DIR / 'outputs' / '2_keypoints_detection' / exp_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if cfg.use_wandb:
        wandb.login()

    indices_dict = split_data(cfg, LMDB_DIR)
    extra_train_info, extra_valid_info = split_extra_data(
        cfg, EXTRA_TRAIN_DIRS, EXTRA_VALID_DIRS)

    for fold in cfg.use_fold:

        if cfg.use_wandb:
            wandb.config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True)
            wandb.init(project=cfg.wandb_project, entity='luka-magic',
                       name=f'{exp_name}', config=wandb.config)
            wandb.config.fold = fold

        train_loader, valid_loader, extra_valid_dict = prepare_dataloader(
            cfg, LMDB_DIR, indices_dict[fold]['train'], indices_dict[fold]['valid'], extra_train_info, extra_valid_info)

        best_score = {
            'loss': float('inf'),
            'accuracy': 0.0
        }

        # model
        if cfg.model_arch == 'hourglassnet':
            model = get_pose_net(cfg.output_size).to(device)
        elif cfg.model_arch == 'pspnet':
            model = PSPNet((cfg.img_h, cfg.img_w)).to(device)
        else:
            NotImplementedError
        print(summary(model, (cfg.input_size, cfg.img_h, cfg.img_w)))

        if cfg.pretrained_model_path is not None:
            model.backbone.load_state_dict(torch.load(
                SAVE_DIR.parent / cfg.pretrained_model_path)['model'], strict=False)

        # loss
        if cfg.loss_fn == 'CenterLoss':
            loss_fn = CenterLoss()
        elif cfg.loss_fn == 'CenterSourceWeightLoss':
            loss_fn = CenterSourceWeightLoss(
                weight_extracted=cfg.weight_extracted)
        else:
            NotImplementedError

        # optimizer
        if cfg.optimizer == 'AdamW':
            optimizer = optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == 'RAdam':
            optimizer = optim.RAdam(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            NotImplementedError

        # scheduler
        if cfg.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.T_0, eta_min=cfg.eta_min)
        elif cfg.scheduler == 'OneCycleLR':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, total_steps=cfg.n_epochs * len(train_loader), max_lr=cfg.lr, pct_start=cfg.pct_start, div_factor=cfg.div_factor, final_div_factor=cfg.final_div_factor)
        else:
            NotImplementedError

        # grad scaler
        scaler = GradScaler(enabled=cfg.use_amp)
        point_counter = PointCounter(cfg)

        for epoch in range(1, cfg.n_epochs + 1):
            wandb_dict = {
                'epoch': epoch,
            }
            valid_accuracy_dict = {}
            train_loss, lr = train_one_epoch(
                cfg, epoch, train_loader, model, loss_fn, device, optimizer, scheduler, cfg.scheduler_step_time, scaler, point_counter)
            valid_acc_per_thr = valid_one_epoch(
                cfg, epoch, valid_loader, model, loss_fn, device, point_counter)

            wandb_dict['lr'] = lr
            wandb_dict['train_loss'] = train_loss
            print('-'*8)
            print(f'Epoch {epoch}/{cfg.n_epochs}')
            print(
                # f'    Train Loss: {train_loss:.5f}, Accuracy: {train_accuracy*100:.2f}%, lr: {lr:.7f}')
                f'    Train Loss: {train_loss:.5f}, lr: {lr:.7f}')
            for thr in thresholds:
                print(
                    f'    Valid Accuracy thr:{thr} => {valid_acc_per_thr[thr].avg*100:.1f}%')
            valid_accuracy_dict['valid'] = valid_acc_per_thr[wandb_thr].avg
            for dataset_name, extra_valid in extra_valid_dict.items():
                extra_valid_acc_per_thr = valid_one_epoch(
                    cfg, epoch, extra_valid['valid_loader'], model, loss_fn, device, point_counter)
                print(f'    EXTRA VALID - {dataset_name}:')
                for thr in thresholds:
                    print(
                        f'    Valid Accuracy thr:{thr} => {extra_valid_acc_per_thr[thr].avg*100:.1f}%')
                valid_accuracy_dict[dataset_name] = extra_valid_acc_per_thr[thr]

            valid_accuracy = np.mean(list(valid_accuracy_dict.values()))
            wandb_dict['valid_accuracy'] = valid_accuracy

            # save model
            save_dict = {
                'epoch': epoch,
                'model': model.state_dict()
            }
            if valid_accuracy > best_score['accuracy']:
                best_score['accuracy'] = valid_accuracy
                torch.save(save_dict, str(SAVE_DIR / 'best_accuracy.pth'))
                if cfg.use_wandb:
                    wandb.run.summary['best_accuracy'] = best_score['accuracy']
            del save_dict
            gc.collect()
            print('-'*80)
            # wandb
            if cfg.use_wandb:
                wandb.log(wandb_dict)
    wandb.finish()
    del model, train_loader, valid_loader, loss_fn, optimizer, scheduler, best_loss, best_accuracy


if __name__ == '__main__':
    main()
