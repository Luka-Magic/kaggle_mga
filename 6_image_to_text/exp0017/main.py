# basic
import sys
import gc
import numpy as np
import pandas as pd
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
from typing import List, Dict, Union, Tuple, Any
import cv2

# hydra
import hydra
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize_config_dir

# wandb
import wandb

# sklearn
from sklearn.model_selection import KFold, StratifiedKFold

# albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# pytorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torchvision.transforms import Compose
from torch.cuda.amp import autocast, GradScaler

# transformers
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from utils import seed_everything, AverageMeter, round_float, is_nan, get_lr
from metrics import validation_metrics


BOS_TOKEN = "<|BOS|>"
X_START = "<|x_start|>"
X_END = "<|x_end|>"
Y_START = "<|y_start|>"
Y_END = "<|y_end|>"

SEPARATOR_TOKENS = [
    BOS_TOKEN,
    X_START,
    X_END,
    Y_START,
    Y_END,
]

LINE_TOKEN = "<line>"
VERTICAL_BAR_TOKEN = "<vertical_bar>"
HORIZONTAL_BAR_TOKEN = "<horizontal_bar>"
SCATTER_TOKEN = "<scatter>"
DOT_TOKEN = "<dot>"

CHART_TYPE_TOKENS = [
    LINE_TOKEN,
    VERTICAL_BAR_TOKEN,
    HORIZONTAL_BAR_TOKEN,
    SCATTER_TOKEN,
    DOT_TOKEN,
]

new_tokens = SEPARATOR_TOKENS + CHART_TYPE_TOKENS

CHART_TYPE2LABEL = {
    'line': 0,
    'vertical_bar': 1,
    'scatter': 2,
    'dot': 3,
    'horizontal_bar': 4
}

pad_token_id = None
best_score = 0.
processor = None
max_length = 1024
max_patches = 1024
n_images = 0

# Data split


def split_data(cfg, lmdb_dir) -> Dict[int, Dict[str, Any]]:
    """
    データからextractedだけを抜き取りkfoldでsplitさせる
    その後trainデータにgeneratedのデータを全て合わせる

    Returns:
        indices_dict (Dict[int, Dict[str, List[int]]]): A dictionary containing train/valid indices of each fold.
            Example: {1: {'train': [0, 1, 4, ...], 'valid': [5, 7, ....], 'gt_df': a dataFrame}}
    """
    indices_dict = {}

    env = lmdb.open(str(lmdb_dir), max_readers=32,
                    readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))

    generated_indicies = []
    extracted_indices = []

    extracted_info = {
        'chart_type': [],
        'id': [],
        'x': [],
        'y': [],
    }
    stratified_label = []
    if cfg.debug:
        n_samples = 5000

    for idx in tqdm(range(n_samples), total=n_samples):
        with env.begin(write=False) as txn:
            # load json
            label_key = f'label-{str(idx+1).zfill(8)}'.encode()
            label = txn.get(label_key).decode('utf-8')
        json_dict = json.loads(label)

        label_source = json_dict['source']

        if label_source == 'extracted':
            extracted_indices.append(idx)
            extracted_info['chart_type'].append(json_dict['chart-type'])
            xs, ys = [], []
            for data_series_dict in json_dict['data-series']:
                xs.append(data_series_dict['x'])
                ys.append(data_series_dict['y'])
            extracted_info['id'].append(json_dict['id'])
            extracted_info['x'].append(xs)
            extracted_info['y'].append(ys)
            stratified_label.append(CHART_TYPE2LABEL[json_dict['chart-type']])
        elif label_source == 'generated':
            generated_indicies.append(idx)

            # cfg.sample_ratio
            # n_generated_indicies = len(generated_indicies)
            generated_indicies = random.sample(
                generated_indicies, len(generated_indicies) * cfg.sample_ratio)

    if cfg.split_method == 'StratifiedKFold':
        for fold, (train_fold_indices, valid_fold_indices) \
                in enumerate(StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed).split(extracted_indices, stratified_label)):
            extracted_fold_info = {
                k: [v[i] for i in valid_fold_indices] for k, v in extracted_info.items()}
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
            gt_df = pd.DataFrame(
                index=[f"{id_}_x" for id_ in extracted_fold_info['id']] +
                [f"{id_}_y" for id_ in extracted_fold_info['id']],
                data={
                    "data_series": extracted_fold_info['x'] + extracted_fold_info['y'],
                    "chart_type": extracted_fold_info['chart_type'] * 2,
                })
            indices_dict[fold] = {
                'train': train_indices,
                'valid': valid_indices,
                'gt_df': gt_df
            }
    else:
        NotImplementedError

    return indices_dict


def get_transforms(cfg, mode='generated'):
    if mode == 'generated':
        return A.Compose([
            # 色彩変換
            A.OneOf([
                A.ChannelShuffle(p=1.),  # RGBの並び替え,
                A.RandomGamma((30, 180), p=1.),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.05, 0.2),
                    contrast_limit=(-0.05, 0.2),
                    p=1.),
            ], p=0.3),

            # gray scale
            A.ToGray(p=0.1),

            # 低画質化
            A.OneOf([
                A.GaussianBlur(
                    blur_limit=(3, 5),
                    sigma_limit=0.7,
                    p=1.),
                A.GaussianBlur(
                    blur_limit=(3, 5),
                    sigma_limit=0.7,
                    p=1.),
                A.GaussNoise(mean=30, p=1.),
                A.JpegCompression(quality_lower=50, quality_upper=100, p=1.),
                A.Downscale(scale_min=0.75, scale_max=0.99, p=1.),
            ], p=0.3),

            A.Resize(height=cfg.img_h, width=cfg.img_w),
            A.Normalize(cfg.img_mean, cfg.img_std),
            ToTensorV2()
        ])
    elif mode == 'extracted':
        return A.Compose([
            A.Resize(height=cfg.img_h, width=cfg.img_w),
            A.Normalize(cfg.img_mean, cfg.img_std),
            ToTensorV2()
        ])

    elif mode == 'valid':
        return A.Compose([
            A.Resize(height=cfg.img_h, width=cfg.img_w),
            A.Normalize(cfg.img_mean, cfg.img_std),
            ToTensorV2()
        ])


# Dataset
class MgaDataset(Dataset):
    def __init__(self, cfg, lmdb_dir, indices, processor, phase):
        self.cfg = cfg
        self.indices = indices
        self.processor = processor
        if phase == 'train':
            self.transforms_gen = get_transforms(cfg, mode='generated')
            self.transforms_ext = get_transforms(cfg, mode='extracted')
        elif phase == 'valid':
            self.transforms = get_transforms(cfg, mode='valid')

        self.phase = phase
        self.env = lmdb.open(str(lmdb_dir), max_readers=32,
                             readonly=True, lock=False, readahead=False, meminit=False)

    def _json_dict_to_gt_string(self, json_dict: Dict[str, Any]) -> str:
        """
        Args:
            json_dict (Dict[str, Any]): ターゲットのdict
        Returns:
            gt_string (str): 入力となるプロンプト
        """
        all_x, all_y = [], []

        for d in json_dict['data-series']:
            x = d["x"]
            y = d["y"]

            x = round_float(x)
            y = round_float(y)

            # Ignore nan values
            if is_nan(x) or is_nan(y):
                continue

            all_x.append(x)
            all_y.append(y)

        chart_type = f"<{json_dict['chart-type']}>"
        x_str = X_START + ";".join(list(map(str, all_x))) + X_END
        y_str = Y_START + ";".join(list(map(str, all_y))) + Y_END

        gt_string = BOS_TOKEN + chart_type + x_str + y_str

        return gt_string, list(map(str, all_x)), list(map(str, all_y))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[int], str]]:
        """
        lmdbからidに一致したimageとlabelを取り出す

        image
            - byteをdecodeしてPIL.Image -> numpyにする

        label
            - byteからjson.loadsでdictにする
                keys: ['source', 'chart-type', 'plot-bb', 'text',
                    'axes', 'data-series', 'id', 'key_point']
            - 'data-series'から正解となるpromptを生成

        Returns:
            samples (Dict[str, Union[torch.Tensor, List[int], str]])
                pixel_values (torch.Tensor): 画像
                input_ids (List[int]): token idのリスト
                ids (str)
        """
        idx = self.indices[idx]
        with self.env.begin(write=False) as txn:
            # load image
            img_key = f'image-{str(idx+1).zfill(8)}'.encode()
            imgbuf = txn.get(img_key)

            # load json
            label_key = f'label-{str(idx+1).zfill(8)}'.encode()
            label = txn.get(label_key).decode('utf-8')

        # label: ['source', 'chart-type', 'plot-bb', 'text', 'axes', 'data-series', 'id', 'key_point']
        json_dict = json.loads(label)

        # image
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        image_arr = np.array(Image.open(buf).convert('RGB'))
        h, w, _ = image_arr.shape

        if self.phase == 'train':
            if json_dict['source'] == 'generated':
                image = self.transforms_gen(image=image_arr)['image']
            elif json_dict['source'] == 'extracted':
                image = self.transforms_ext(image=image_arr)['image']
        else:
            image = self.transforms(image=image_arr)['image']
        encoding = {}
        encoding['image_tensor'] = image

        gt_string, x_list, y_list = self._json_dict_to_gt_string(json_dict)

        encoding['text'] = gt_string
        encoding['id'] = json_dict['id']
        encoding['phase'] = self.phase
        if self.phase == 'valid':
            encoding['info'] = {
                'img': image_arr,
                'img_h': h,
                'img_w': w,
                'source': json_dict['source'],
                'x_tick_type': json_dict['axes']['x-axis']['tick-type'],
                'y_tick_type': json_dict['axes']['y-axis']['tick-type'],
                'gt_x': x_list,
                'gt_y': y_list,
                'chart_type': json_dict['chart-type']
            }
        return encoding

# Collate_fn


def collate_fn(samples: List[Dict[str, Union[torch.Tensor, List[int], str]]]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    Returns:
        batch (dict):
            keys: (flattened_patches, attention_mask, labels, id)
    """

    texts = [item['text'] for item in samples]
    images = [item['image_tensor'] for item in samples]

    batch = processor(
        images=images,
        random_padding=True,
        add_special_tokens=True,
        max_patches=max_patches,
        return_tensors='pt'
    )

    phase = samples[0]['phase']

    # Make a multiple of 8 to efficiently use the tensor cores
    text_inputs = processor(
        text=texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=max_length
    )
    batch['labels'] = text_inputs.input_ids

    batch["id"] = [x["id"] for x in samples]
    if phase == 'valid':
        batch['info'] = [x['info'] for x in samples]
    return batch


# Dataloader


def prepare_dataloader(cfg, lmdb_dir, processor, train_indices, valid_indices):
    train_ds = MgaDataset(cfg, lmdb_dir, train_indices,
                          processor, 'train')
    valid_ds = MgaDataset(cfg, lmdb_dir, valid_indices,
                          processor, 'valid')
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.valid_bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return train_loader, valid_loader


# Train function
def train_valid_one_epoch(
    cfg,
    fold: int,
    epoch: int,
    save_dir: Path,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    processor: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    scheduler_step_time: str,
    scaler: torch.cuda.amp.GradScaler,
    gt_df: pd.DataFrame
):
    global best_score, n_images

    model.train()

    train_losses = AverageMeter()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    valid_count_per_epoch = 0

    # train & valid
    for step, batch in pbar:
        step += 1
        flattened_patches = batch['flattened_patches'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch["labels"].to(device)
        bs = len(flattened_patches)
        n_images += bs
        with autocast(enabled=cfg.use_amp):
            output = model(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = output.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        train_losses.update(loss.item(), bs)
        lr = get_lr(optimizer)
        if scheduler is not None and scheduler_step_time == 'step':
            scheduler.step()
        pbar.set_description(
            f'[TRAIN epoch {epoch}/{cfg.n_epochs} ({valid_count_per_epoch}/{cfg.n_valid_per_epoch})]')
        pbar.set_postfix(OrderedDict(loss=train_losses.avg))
        # if cfg.use_wandb:
        #     wandb.log({
        #         'n_images': n_images,
        #         'train_loss': loss.item(),
        #         'lr': lr
        #     })

        if step % (len(train_loader) // cfg.n_valid_per_epoch) == 0:
            # valid
            valid_score = valid_function(cfg, epoch, valid_loader,
                                         processor, model, device, gt_df)
            model.train()
            valid_count_per_epoch += 1
            print("\n" + "=" * 80)
            print(
                f'Fold {fold} | Epoch {epoch}/{cfg.n_epochs} ({valid_count_per_epoch}/{cfg.n_valid_per_epoch})')
            print(f'    TRAIN:')
            print(f'            loss: {train_losses.avg:.6f}')
            print(f'    VALID:')
            for valid_score_name, valid_score_value in valid_score.items():
                print(
                    f'            {valid_score_name}: {valid_score_value:.6f}')
            print("=" * 80)

            # log
            if cfg.use_wandb:
                wandb_log = {
                    'n_images': n_images,
                    'epoch': epoch,
                    'train_loss': train_losses.avg,
                    'lr': lr,
                }
                wandb_log.update(valid_score)
                wandb.log(
                    wandb_log
                )
            # save model
            if valid_score['valid_score'] > best_score:
                best_score = valid_score['valid_score']
                model.save_pretrained(str(save_dir))
                processor.save_pretrained(
                    str(save_dir))
                with open(save_dir / 'best_score_info.json', 'w') as f:
                    save_dict = {str(fold): {
                        'epoch': epoch,
                        'n_images': n_images,
                        'best_score': best_score
                    }}
                    json.dump(save_dict, f)
                if cfg.use_wandb:
                    wandb.run.summary['best_score'] = best_score
                gc.collect()
                torch.cuda.empty_cache()
        if scheduler is not None and scheduler_step_time == 'epoch':
            scheduler.step()


def valid_function(
    cfg,
    epoch: int,
    dataloader: DataLoader,
    processor: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    gt_df: pd.DataFrame
) -> Dict[str, float]:
    global n_images

    model.eval()

    outputs = []
    ids = []
    table_info_list = []

    for step, batch in enumerate(dataloader):
        flattened_patches = batch['flattened_patches'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        bs = len(flattened_patches)
        decoder_input_ids = torch.full(
            (bs, 1),
            model.config.decoder_start_token_id,
            device=device,
        )

        output = model.generate(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True
        )

        outputs.extend(processor.tokenizer.batch_decode(output.sequences))
        ids.extend(batch['id'])
        for info in batch['info']:
            table_info_list.append(info)

    scores, pred_list = validation_metrics(outputs, ids, gt_df)
    return scores


# main
def main():
    EXP_PATH = Path.cwd()
    with initialize_config_dir(config_dir=str(EXP_PATH / 'config')):
        cfg = compose(config_name='config.yaml')

    ROOT_DIR = Path.cwd().parents[2]
    exp_name = EXP_PATH.name
    project_name = Path.cwd().parent.stem
    LMDB_DIR = ROOT_DIR / 'data' / cfg.dataset_name / 'lmdb'
    SAVE_DIR = ROOT_DIR / 'outputs' / project_name / exp_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)

    if cfg.use_wandb:
        wandb.login()

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    indices_per_fold = split_data(cfg, LMDB_DIR)

    for fold in cfg.use_fold:
        if cfg.use_wandb:
            wandb.config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True)
            wandb.init(project=cfg.wandb_project, entity='luka-magic',
                       name=f'{exp_name}', config=wandb.config)
            wandb.config.fold = fold

        # restart or load pretrained model from internet
        pretrained_path = SAVE_DIR.parent / cfg.pretrained_model_exp_name if cfg.restart \
            else cfg.pretrained_model_from_net_path

        # TODO: save dirにrestartで取ってこれるようにepochやbest scoreをjsonで保存するように実装
        # config
        global max_length, max_patches, processor, new_tokens, pad_token_id, n_images, best_score
        max_length = cfg.max_length
        max_patches = cfg.max_patches
        # processor
        processor = AutoProcessor.from_pretrained(str(pretrained_path))
        processor.image_processor.size = {
            "height": cfg.img_h,
            "width": cfg.img_w,
        }
        processor.image_processor.is_vqa = False
        processor.tokenizer.add_tokens(new_tokens)
        pad_token_id = processor.tokenizer.pad_token_id

        # model
        model = Pix2StructForConditionalGeneration.from_pretrained(
            str(pretrained_path)).to(device)

        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        model.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([
                                                                                 BOS_TOKEN])[0]
        model.config.text_config.is_decoder = True

        print(f'load model: {str(pretrained_path)}')
        if cfg.restart:
            print(f'------------ Restart Learning ------------')
        else:
            print('------------Start Learning------------')

        # save
        model.save_pretrained(str(SAVE_DIR))
        processor.save_pretrained(
            str(SAVE_DIR))

        # data
        train_indices, valid_indices = indices_per_fold[fold]['train'], indices_per_fold[fold]['valid']
        train_loader, valid_loader = prepare_dataloader(
            cfg, LMDB_DIR, processor, train_indices, valid_indices)

        # optimizer
        optimizer = Adafactor(model.parameters(
        ), scale_parameter=False, relative_step=False, lr=cfg.lr, weight_decay=cfg.weight_decay)

        # scaler
        scaler = GradScaler(enabled=cfg.use_amp)

        # scheduelr
        if cfg.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.T_0, eta_min=cfg.eta_min)
        elif cfg.scheduler == 'OneCycleLR':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, total_steps=cfg.n_epochs * len(train_loader), max_lr=cfg.lr, pct_start=cfg.pct_start, div_factor=cfg.div_factor, final_div_factor=cfg.final_div_factor)
        elif cfg.scheduler == 'huggingface_scheduler':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.warmup_step, num_training_steps=cfg.n_epochs * len(train_loader))
        else:
            scheduler = None

        if cfg.restart:
            with open(pretrained_path / 'best_score_info.json', 'r') as f:
                best_score_dict = json.load(f)
                start_epoch = best_score_dict[str(fold)]['epoch'] + 1
                n_images = best_score_dict[str(fold)]['n_images']
                best_score = best_score_dict[str(fold)]['best_score']
        else:
            n_images = 0
            start_epoch = 1
            best_score = 0.0

        for epoch in range(start_epoch, cfg.n_epochs + 1):
            train_valid_one_epoch(
                cfg, fold, epoch, SAVE_DIR, train_loader, valid_loader, processor, model, device, optimizer, scheduler, cfg.scheduler_step_time, scaler, indices_per_fold[fold]['gt_df'])
    wandb.finish()
    del model, processor, train_loader, valid_loader, train_indices, valid_indices, optimizer, scaler


if __name__ == '__main__':
    main()
