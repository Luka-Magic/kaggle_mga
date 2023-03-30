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
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torchvision.transforms import Compose
from torch.cuda.amp import autocast, GradScaler

# other
import albumentations
from albumentations import KeypointParams
from albumentations.pytorch import ToTensorV2

from utils import seed_everything, AverageMeter, calc_accuracy, get_final_preds
from pose_resnet import get_pose_net
from loss import JointsMSELoss


def split_data(cfg, lmdb_dir):
    indices_dict = {}

    env = lmdb.open(str(lmdb_dir), max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))
    
    labels = []
    indices = []
    # check data
    for idx in tqdm(range(n_samples), total=n_samples):
        with env.begin(write=False) as txn:
            # load json
            label_key = f'label-{str(idx+1).zfill(8)}'.encode()
            label = txn.get(label_key).decode('utf-8')
        json_dict = json.loads(label)
        try:
            joints = np.array([[d['x'], d['y']] for d in json_dict['key_point']])
        except:
            continue
        if len(joints) == 0:
            continue
        indices.append(idx)

    print('num-samples: ', len(indices))

    if cfg.split_method == 'KFold':
        for fold, (train_fold_indices, vaild_fold_indices) \
                in enumerate(KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed).split(indices)):
            indices_dict[fold] = {
                'train': [indices[i] for i in train_fold_indices],
                'valid': [indices[i] for i in vaild_fold_indices]
            }
    elif cfg.split_method == 'StratifiedKFold':
        for idx in indices:
            with env.begin(write=False) as txn:
                # load json
                label_key = f'label-{str(idx+1).zfill(8)}'.encode()
                label = txn.get(label_key).decode('utf-8')
            json_dict = json.loads(label)
            label = cfg.chart_type2label[json_dict['chart-type']]
            labels.append(label)
        for fold, (train_fold_indices, vaild_fold_indices) \
                in enumerate(StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed).split(indices, labels)):
            indices_dict[fold] = {
                'train': [indices[i] for i in train_fold_indices],
                'valid': [indices[i] for i in vaild_fold_indices]
            }
    return indices_dict

# Lmdb Dataset
class MgaLmdbDataset(Dataset):
    def __init__(self, cfg, lmdb_dir, indices, transforms):
        super().__init__()
        self.cfg = cfg
        self.transforms = transforms
        self.indices = indices
        self.env = lmdb.open(str(lmdb_dir), max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        self.n_joints = cfg.output_size
        self.sigma = cfg.sigma
        self.img_h, self.img_w = cfg.img_h, cfg.img_w
        self.heatmap_h, self.heatmap_w = cfg.heatmap_h, cfg.heatmap_w

    def _create_heatmap(self, joints):
        '''
            joints: [(x1, y1), (x2, y2), ...]
            heatmap: size: (n_joints, hm_h, hm_w)
        '''
        heatmap = np.zeros((self.n_joints, self.heatmap_h, self.heatmap_w), dtype=np.float32)
        for joint_id in range(len(joints)):
            mu_x = joints[joint_id][0]
            mu_y = joints[joint_id][1]
            
            x = np.arange(0, self.heatmap_w, 1, np.float32)
            y = np.arange(0, self.heatmap_h, 1, np.float32)
            y = y[:, np.newaxis]

            heatmap[joint_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))
        return heatmap
    
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
        
        # label
        json_dict = json.loads(label)
        keypoints = [[dic['x'], dic['y']] for dic in json_dict['key_point']]
        kp_arr = np.array(keypoints)
        kp_min = np.amin(kp_arr, 0)
        if kp_min[0] < 0 or kp_min[1] < 0:
            # print(keypoints)
            print(json_dict['id'])

        transformed = self.transforms(image=img, keypoints=keypoints)
        img = transformed['image']
        keypoints = transformed['keypoints']
        keypoints_on_hm = np.array(keypoints) * \
            np.array([self.heatmap_w, self.heatmap_h]) / np.array([self.img_w, self.img_h])

        heatmap_weight = np.zeros(self.n_joints, dtype=np.int32)
        heatmap_weight[:len(keypoints)] = 1

        heatmap = self._create_heatmap(keypoints_on_hm)

        img = torch.from_numpy(img).permute(2, 0, 1)
        heatmap = torch.from_numpy(heatmap)
        heatmap_weight = torch.from_numpy(heatmap_weight)

        return img, heatmap, heatmap_weight


def get_transforms(cfg, phase):
    if phase == 'train':
        aug = cfg.train_aug
    elif phase == 'valid':
        aug = cfg.valid_aug
    elif phase == 'tta':
        aug = cfg.tta_aug

    augs = [getattr(albumentations, name)(**kwargs) for name, kwargs in aug.items()]
    # augs.append(ToTensorV2(p=1.))
    return albumentations.Compose(augs, keypoint_params=KeypointParams(format='xy'))


def prepare_dataloader(cfg, lmdb_dir, train_indices, valid_indices):
    train_ds = MgaLmdbDataset(cfg, lmdb_dir, train_indices,
                          transforms=get_transforms(cfg, 'train'))
    valid_ds = MgaLmdbDataset(cfg, lmdb_dir, valid_indices,
                          transforms=get_transforms(cfg, 'valid'))
    valid_tta_ds = MgaLmdbDataset(
        cfg, lmdb_dir, valid_indices, transforms=get_transforms(cfg, 'tta'))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.valid_bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    valid_tta_loader = DataLoader(
        valid_tta_ds,
        batch_size=cfg.valid_bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    return train_loader, valid_loader, valid_tta_loader


def train_one_epoch(cfg, epoch, dataloader, model, loss_fn, device, optimizer, scheduler, scheduler_step_time, scaler):
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    model.train()

    accuracy = AverageMeter()
    losses = AverageMeter()
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for step, (images, heatmaps, heatmap_weight) in pbar:
        images = images.to(device).float()
        heatmaps = heatmaps.to(device).float()
        heatmap_weight = heatmap_weight.to(device).long()
        bs = len(images)

        with autocast(enabled=cfg.use_amp):
            pred = model(images)
            loss = loss_fn(pred, heatmaps, heatmap_weight)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        _, avg_acc, cnt, pred = calc_accuracy(pred.detach().cpu().numpy(),
                                              heatmaps.detach().cpu().numpy())
        accuracy.update(avg_acc, cnt)
        losses.update(loss.item(), bs)
        lr =  get_lr(optimizer)
        if scheduler_step_time == 'step':
            scheduler.step()
        pbar.set_description(f'[Train epoch {epoch}/{cfg.n_epochs}]')
        pbar.set_postfix(OrderedDict(loss=losses.avg, accuracy=accuracy.avg))
        if cfg.use_wandb:
            wandb.log({
                'step': (epoch - 1) * len(pbar) + step,
                'train_accuracy': accuracy.avg,
                'train_loss': losses.avg,
                'lr': lr
            })
    if scheduler_step_time == 'epoch':
        scheduler.step()
    
    lr = get_lr(optimizer)

    return losses.avg, accuracy.avg, lr


def valid_one_epoch(cfg, epoch, dataloader, model, loss_fn, device):
    model.eval()

    accuracy = AverageMeter()
    losses = AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for _, (images, heatmaps, heatmap_weight) in pbar:
        images = images.to(device).float()
        heatmaps = heatmaps.to(device).float()
        heatmap_weight = heatmap_weight.to(device).long()
        bs = len(images)

        with torch.no_grad():
            pred = model(images)
            loss = loss_fn(pred, heatmaps, heatmap_weight)

        _, avg_acc, cnt, pred = calc_accuracy(pred.detach().cpu().numpy(),
                                              heatmaps.detach().cpu().numpy())
        accuracy.update(avg_acc, cnt)
        losses.update(loss.item(), bs)
        
        pbar.set_description(f'[Valid epoch {epoch}/{cfg.n_epochs}]')
        pbar.set_postfix(OrderedDict(loss=losses.avg, accuracy=accuracy.avg))
    
    return losses.avg, accuracy.avg


def main():
    EXP_PATH = Path.cwd()
    with initialize_config_dir(config_dir=str(EXP_PATH / 'config')):
        cfg = compose(config_name='config.yaml')

    ROOT_DIR = Path.cwd().parents[2]
    exp_name = EXP_PATH.name
    LMDB_DIR = ROOT_DIR / 'data' / cfg.dataset_name / 'lmdb'
    SAVE_DIR = ROOT_DIR / 'outputs' / '2_keypoints_detection' / exp_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if cfg.use_wandb:
        wandb.login()

    indices_dict = split_data(cfg, LMDB_DIR)

    for fold in cfg.use_fold:

        if cfg.use_wandb:
            wandb.config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True)
            wandb.init(project=cfg.wandb_project, entity='luka-magic',
                        name=f'{exp_name}', config=wandb.config)
            wandb.config.fold = fold
        
        train_loader, valid_loader, _ = prepare_dataloader(cfg, LMDB_DIR, indices_dict[fold]['train'], indices_dict[fold]['valid'])

        best_score = {
            'loss': float('inf'),
            'accuracy': 0.0
        }

        # model
        if cfg.model_arch == 'hourglassnet':
            model = get_pose_net(cfg.output_size).to(device)
        else:
            NotImplementedError
        print(summary(model, (3, 300, 500)))

        # loss
        if cfg.loss_fn == 'JointsMSELoss':
            loss_fn = JointsMSELoss()
        else:
            NotImplementedError
        
        # optimizer
        if cfg.optimizer == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == 'RAdam':
            optimizer = optim.RAdam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
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

        for epoch in range(1, cfg.n_epochs + 1):
            train_loss, train_accuracy, lr = train_one_epoch(cfg, epoch, train_loader, model, loss_fn, device, optimizer, scheduler, cfg.scheduler_step_time, scaler)
            valid_loss, valid_accuracy =  valid_one_epoch(cfg, epoch, valid_loader, model, loss_fn, device)
            print('-'*80)
            print(f'Epoch {epoch}/{cfg.n_epochs}')
            print(f'    Train Loss: {train_loss:.5f}, Train acc: {train_accuracy*100:.3f}%, lr: {lr:.7f}')
            print(f'    Valid Loss: {valid_loss:.5f}, Valid acc: {valid_accuracy*100:.3f}%')
            print('-'*80)
        
            # save model
            save_dict = {
                'epoch': epoch,
                'valid_loss': valid_loss,
                'valid_accuracy': valid_accuracy,
                'model': model.state_dict()
            }
            if valid_loss < best_score['loss']:
                best_score['loss'] = valid_loss
                torch.save(save_dict, str(SAVE_DIR / 'best_loss.pth'))
                if cfg.use_wandb:
                    wandb.run.summary['best_loss'] = best_score['loss']
            if valid_accuracy > best_score['accuracy']:
                best_score['accuracy'] = valid_accuracy
                torch.save(save_dict, str(SAVE_DIR / 'best_accuracy.pth'))
                if cfg.use_wandb:
                    wandb.run.summary['best_accuracy'] = best_score['accuracy']
            del save_dict
            gc.collect()

            # wandb
            if cfg.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'lr': lr,
                    'valid_loss': valid_loss,
                    'valid_accuracy': valid_accuracy
                })
    wandb.finish()
    del model, train_loader, valid_loader, loss_fn, optimizer, scheduler, best_loss, best_accuracy


        
if __name__ == '__main__':
    main()