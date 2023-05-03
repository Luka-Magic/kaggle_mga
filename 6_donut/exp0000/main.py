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
from polyleven import levenshtein
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
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torchvision.transforms import Compose
from torch.cuda.amp import autocast, GradScaler

# transformers
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
    get_scheduler
)
from transformers import PreTrainedTokenizerBase, PreTrainedModel

# other
# import albumentations
# from albumentations.pytorch import ToTensorV2

from utils import seed_everything, AverageMeter, round_float, is_nan
from metrics import validation_metrics

PROMPT_TOKEN = "<|PROMPT|>"
X_START = "<x_start>"
X_END = "<x_end>"
Y_START = "<y_start>"
Y_END = "<y_end>"

SEPARATOR_TOKENS = [
    PROMPT_TOKEN,
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
unk_token_id = None

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
        'x': [],
        'y': [],
    }
    stratified_label = []

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
            extracted_info['x'].append(xs)
            extracted_info['y'].append(ys)
            stratified_label.append(CHART_TYPE2LABEL[json_dict['chart-type']])
        elif label_source == 'generated':
            generated_indicies.append(idx)

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
            indices_dict[fold] = {
                'train': train_indices,
                'valid': valid_indices,
                'gt_df': pd.DataFrame(
                    index=[f"{id_}_x" for id_ in valid_indices] +
                        [f"{id_}_y" for id_ in valid_indices],
                    data={
                        "data_series": extracted_fold_info['x'] + extracted_fold_info['y'],
                        "chart_type": extracted_fold_info['chart_type'] * 2,
                    },
                )
            }
    else:
        NotImplementedError

    return indices_dict


# Dataset
class MgaDataset(Dataset):
    def __init__(self, cfg, lmdb_dir, indices, processor, output=True):
        self.cfg = cfg
        self.indices = indices
        self.processor = processor
        self.output = output
        self.env = lmdb.open(str(lmdb_dir), max_readers=32,
                             readonly=True, lock=False, readahead=False, meminit=False)
        # TODO: test時の処理が必要か考える

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

        gt_string = PROMPT_TOKEN + chart_type + x_str + y_str

        return gt_string

    def _replace_unk_tokens_with_one(self, example_ids: List[int], example_tokens: List[str], one_token_id: int, unk_token_id: int) -> List[int]:
        """
        <unk>でかつテキストが1のものを<one>のidにして返す。

        Args:
            example_ids (list): tokenのid
            example_tokens (list): tokenのテキスト
            one_token_id (int): <one>のid
            unk_token_id (int): <unk>のid

        Returns:
            list: 1を<one>のidに変換させたlist
        """
        replaced_ids = []
        for id_, token in zip(example_ids, example_tokens):
            if id_ == unk_token_id and token == "1":
                id_ = one_token_id
            replaced_ids.append(id_)
        return replaced_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[int], str]]:
        """

        lmdbからidに一致したimageとlabelを取り出す

        image
            - byteをdecodeしてPIL.Image -> numpyにする

        label
            - byteからjson.loadsでdictにする
                keys: ['source', 'chart-type', 'plot-bb', 'text', 'axes', 'data-series', 'id', 'key_point']
            - 'data-series'からprocessor.tokenizerでid化
            - processor.tokenizer.tokenizeでテキストで<unk>のものが出る
            - idが<unk>のidでかつtextが'1'である場所を<one>のidに変換する。このidを最終的なidとする

        中間にこの変数が必要？
            keys: (ground_truth, x, y, chart-type, id, source, image_path, width, height, unk_tokens)

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

        # image
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        pixel_values = self.processor(
            images=Image.open(buf).convert('RGB'),
            random_padding=True
        ).pixel_values[0]

        # label: ['source', 'chart-type', 'plot-bb', 'text', 'axes', 'data-series', 'id', 'key_point']
        json_dict = json.loads(label)

        gt_string = self._json_dict_to_gt_string(json_dict)

        ids = self.processor.tokenizer(
            gt_string,
            add_special_tokens=False,
            max_length=self.cfg.max_length,
            padding=True,
            truncation=True
        ).input_ids

        tokens = self.processor.tokenizer.tokenize(
            gt_string, add_special_tokens=False)

        one_token_id = self.processor.tokenizer(
            '<one>', add_special_tokens=False).input_ids[0]
        unk_token_id = self.processor.tokenizer.unk_token_id
        final_ids = self._replace_unk_tokens_with_one(
            ids, tokens, one_token_id, unk_token_id)

        return {
            'pixel_values': torch.tensor(pixel_values),
            'input_ids': final_ids,
            'id': idx,
        }

# Collate_fn


def collate_fn(samples: List[Dict[str, Union[torch.Tensor, List[int], str]]]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    batch = {}

    batch['pixel_values'] = torch.stack([x['pixel_values'] for x in samples])

    max_length = max([len(x["input_ids"]) for x in samples])

    # Make a multiple of 8 to efficiently use the tensor cores
    if max_length % 8 != 0:
        max_length = (max_length // 8 + 1) * 8

    input_ids = [
        x["input_ids"] + [pad_token_id] * (max_length - len(x["input_ids"]))
        for x in samples
    ]

    labels = torch.tensor(input_ids)
    # TODO: 何の意味があるか調べる
    # ignore loss on padding tokens
    labels[labels == pad_token_id] = -100
    batch["labels"] = labels

    batch["id"] = [x["id"] for x in samples]
    return batch

# Dataloader


def prepare_dataloader(cfg, lmdb_dir, processor, train_indices, valid_indices):
    train_ds = MgaDataset(cfg, lmdb_dir, train_indices, processor)
    valid_ds = MgaDataset(cfg, lmdb_dir, valid_indices, processor)

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


def train_one_epoch(
    cfg,
    epoch: int,
    dataloader: DataLoader,
    processor: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler
) -> float:
    model.train()

    losses = AverageMeter()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for _, batch in pbar:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch["labels"].to(device)
        bs = len(pixel_values)
        with autocast(enabled=cfg.use_amp):
            output = model(
                pixel_values=pixel_values,
                labels=labels
            )
            loss = output.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        losses.update(loss.item(), bs)

        pbar.set_description(f'[TRAIN epoch {epoch} / {cfg.n_epochs}]')
        pbar.set_postfix(OrderedDict(loss=losses.avg))
        if cfg.use_wandb:
            wandb.log({
                'loss': loss.item()
            })
    return losses.avg


def valid_one_epoch(
    cfg,
    epoch: int,
    dataloader: DataLoader,
    processor: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    gt_df: pd.DataFrame
) -> Dict[str, float]:
    model.eval()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    outputs = []
    ids = []

    for _, batch in pbar:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch["labels"].to(device)
        bs = len(pixel_values)
        decoder_input_ids = torch.full(
            (bs, 1),
            model.config.decoder_start_token_id,
            device=device,
        )

        output = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=cfg.max_length,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            top_k=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True
        )

        outputs.extend(processor.tokenizer.batch_decode(output.sequences))
        ids.extend(batch['id'])

        pbar.set_description(f'[VALID epoch {epoch} / {cfg.n_epochs}]')

    metrics = validation_metrics(outputs, ids, gt_df)

    return metrics

# main


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

    seed_everything(cfg.seed)
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
        pretrained_path = cfg.pretrained_model_dir if cfg.restart \
            else cfg.pretrained_model_from_net_path

        # best score
        best_score = 0.

        # model config
        config = VisionEncoderDecoderConfig.from_pretrained(
            cfg.pretrained_model_from_net_path)
        config.encoder.image_size = (cfg.img_h, cfg.img_w)
        config.decoder.max_length = cfg.max_length

        # processor
        processor = DonutProcessor.from_pretrained(pretrained_path)
        processor.image_processor.size = {
            "height": cfg.img_h,
            "width": cfg.img_w,
        }
        pad_token_id = processor.tokenizer.pad_token_id
        config.pad_token_id = processor.tokenizer.pad_token_id
        config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([
                                                                                  PROMPT_TOKEN])[0]

        # model
        model = VisionEncoderDecoderModel.from_pretrained(
            pretrained_path).to(device)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        model.config.pad_token_id = pad_token_id
        model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([
                                                                                        PROMPT_TOKEN])[0]
        # data
        train_indices, valid_indices = indices_per_fold[fold]['train'], indices_per_fold[fold]['valid']
        train_loader, valid_loader = prepare_dataloader(
            cfg, LMDB_DIR, processor, train_indices, valid_indices)

        # optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # scaler
        scaler = GradScaler(enabled=cfg.use_amp)

        for epoch in range(1, cfg.n_epochs + 1):
            train_loss = train_one_epoch(
                cfg, epoch, train_loader, processor, model, device, optimizer, scaler)
            valid_score = valid_one_epoch(
                cfg, epoch, valid_loader, processor, model, device, indices_per_fold[fold]['gt_df'])
            print("=" * 80)
            print(f'Fold {fold} | Epoch {epoch} / {cfg.n_epochs}')
            print(f'    TRAIN: loss: {train_loss:.6f}')
            for valid_score_name, valid_score_value in valid_score.items():
                print(
                    f'            {valid_score_name}: {valid_score_value:.6f}')
            print("=" * 80)

            if cfg.use_wandb:
                wandb_log = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                }
                wandb_log.update(valid_score)
                wandb.log(
                    wandb_log
                )
            if valid_score['valid_score'] > best_score:
                best_score = valid_score['valid_score']
                model.save_pretrained(str(SAVE_DIR / 'best_score.pth'))
                model.processor.save_pretrained(
                    str(SAVE_DIR / 'best_score.pth'))
                if cfg.use_wandb:
                    wandb.run.summary['best_score'] = best_score
    wandb.finish()
    del model, processor, config, train_loader, valid_loader, train_indices, valid_indices, optimizer, scaler


if __name__ == '__main__':
    main()
