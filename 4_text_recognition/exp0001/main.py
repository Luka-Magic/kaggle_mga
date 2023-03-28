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

# hydra
import hydra
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize_config_dir

# wandb
import wandb

# sklearn
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

# pytorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torchvision.transforms import Compose
from torch.cuda.amp import autocast, GradScaler

# other
from torchsummary import summary
import albumentations
from albumentations.pytorch import ToTensorV2

from model import CRNN
from utils import seed_everything, AverageMeter, CTCLabelConverter
from dataset import MgaLmdbDataset, AlignCollate


def split_data(cfg, lmdb_dir):
    indices_dict = {}

    env = lmdb.open(str(lmdb_dir), max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))
    
    indices = list(range(n_samples))

    if  cfg.split_method == 'KFold':
        for fold, (train_fold_indices, vaild_fold_indices) \
                in enumerate(KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed).split(indices)):
            indices_dict[fold] = {
                'train': train_fold_indices,
                'valid': vaild_fold_indices
            }
    elif  cfg.split_method == 'GroupKFold':
        groups = []
        for idx in range(n_samples):
            with env.begin(write=False) as txn:
                idx += 1
                # load json
                label_key = f'label-{str(idx).zfill(8)}'.encode()
                label = txn.get(label_key).decode('utf-8')
            json_dict = json.loads(label)
            id_ = json_dict['id']
            groups.append(id_)
        for fold, (train_fold_indices, vaild_fold_indices) \
                in enumerate(GroupKFold(n_splits=cfg.n_folds).split(indices, groups=groups)):
            indices_dict[fold] = {
                'train': train_fold_indices,
                'valid': vaild_fold_indices
            }
    elif cfg.split_method == 'StratifiedKFold':
        labels = []
        for idx in range(n_samples):
            with env.begin(write=False) as txn:
                idx += 1
                # load json
                label_key = f'label-{str(idx).zfill(8)}'.encode()
                label = txn.get(label_key).decode('utf-8')
            json_dict = json.loads(label)
            label = cfg.chart_type2label[json_dict['chart-type']]
            labels.append(label)
        for fold, (train_fold_indices, vaild_fold_indices) \
                in enumerate(StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed).split(indices, labels)):
            indices_dict[fold] = {
                'train': train_fold_indices,
                'valid': vaild_fold_indices
            }
    return indices_dict


def prepare_dataloader(cfg, lmdb_dir, train_indices, valid_indices):
    train_ds = MgaLmdbDataset(cfg, lmdb_dir, train_indices)
    valid_ds = MgaLmdbDataset(cfg, lmdb_dir, valid_indices)
    train_align_collate = AlignCollate(cfg.img_h, cfg.img_w, cfg.padding, is_valid=False)
    valid_align_collate = AlignCollate(cfg.img_h, cfg.img_w, cfg.padding, is_valid=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=train_align_collate,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.valid_bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=valid_align_collate,
        pin_memory=True
    )
    return train_loader, valid_loader


def train_one_epoch(cfg, epoch, dataloader, converter, model, loss_fn, device, optimizer, scheduler, scheduler_step_time, scaler):
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    model.train()

    accuracy = AverageMeter()
    losses = AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for _, (images, labels) in pbar:
        images = images.to(device).float()
        labels = labels.to(device).long() # ?
        text, length = converter.encode(labels, batch_max_length)
        bs = len(images)

        with autocast(enabled=cfg.use_amp):
            pred = model(images)
            loss = loss_fn(pred, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        pred_labels = torch.argmax(pred.detach().cpu(), dim=1)
        batch_accuracy = (pred_labels == labels.detach().cpu()).sum().item() / bs
        accuracy.update(batch_accuracy, bs)
        losses.update(loss.item(), bs)
        
        if scheduler_step_time == 'step':
            scheduler.step()
        pbar.set_description(f'[Train epoch {epoch}/{cfg.n_epochs}]')
        pbar.set_postfix(OrderedDict(loss=losses.avg, accuracy=accuracy.avg))
    if scheduler_step_time == 'epoch':
        scheduler.step()
    
    lr = get_lr(optimizer)

    return losses.avg, accuracy.avg, lr


def valid_one_epoch(cfg, epoch, dataloader, converter, model, loss_fn, device):
    model.eval()

    accuracy = AverageMeter()
    losses = AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for _, (images, texts) in pbar:
        images = images.to(device).float()
        labels = labels.to(device).long()
        text_encodes, lengths = converter.encode(texts, cfg.batch_max_length)
        bs = len(images)

        with torch.no_grad():
            pred = model(images, text_encodes) # (bs, n_chars, 512)
            pred = pred.log_softmax(2).permute(1, 0, 2)
            loss = loss_fn(pred, text_encodes)

        pred_labels = torch.argmax(pred.detach().cpu(), dim=1)
        batch_accuracy = (pred_labels == labels.detach().cpu()).sum().item() / bs
        accuracy.update(batch_accuracy, bs)
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
    SAVE_DIR = ROOT_DIR / 'outputs' / exp_name
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
        model = CRNN(cfg)

        # loss
        if cfg.loss_fn == 'CTCLoss':
            loss_fn = nn.CTCLoss().to(device)
        
        # optimizer
        if cfg.optimizer == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == 'RAdam':
            optimizer = optim.RAdam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            NotImplementedError
        
        # scheduler
        # if cfg.scheduler == 'CosineAnnealingWarmRestarts':
        #     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #         optimizer, T_0=cfg.T_0, eta_min=cfg.eta_min)
        # elif cfg.scheduler == 'OneCycleLR':
        #     scheduler = optim.lr_scheduler.OneCycleLR(
        #         optimizer, total_steps=cfg.n_epochs * len(train_loader), max_lr=cfg.lr, pct_start=cfg.pct_start, div_factor=cfg.div_factor, final_div_factor=cfg.final_div_factor)
        # else:
        #     NotImplementedError
        
        # grad scaler
        scaler = GradScaler(enabled=cfg.use_amp)

        # CTC label converter
        converter = CTCLabelConverter(character) # create characterしなきゃ

        for epoch in range(1, cfg.n_epochs + 1):
            train_loss, train_accuracy, lr = train_one_epoch(cfg, epoch, train_loader, converter, model, loss_fn, device, optimizer, scheduler, cfg.scheduler_step_time, scaler)
            valid_loss, valid_accuracy =  valid_one_epoch(cfg, epoch, valid_loader, converter, model, loss_fn, device)
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