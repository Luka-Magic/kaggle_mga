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
X_START = "<x_start>"
X_END = "<x_end>"
Y_START = "<y_start>"
Y_END = "<y_end>"

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

    if cfg.split_method == 'StratifiedKFold':
        for fold, (_, valid_fold_indices) \
                in enumerate(StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed).split(extracted_indices, stratified_label)):
            extracted_fold_info = {
                k: [v[i] for i in valid_fold_indices] for k, v in extracted_info.items()}

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
                'valid': valid_indices,
                'gt_df': gt_df
            }
    else:
        NotImplementedError

    return indices_dict


# Dataset
class MgaDataset(Dataset):
    def __init__(self, cfg, lmdb_dir, indices, processor, transforms, phase):
        self.cfg = cfg
        self.indices = indices
        self.processor = processor
        self.phase = phase
        self.transforms = transforms
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

        # image
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        image_arr = np.array(Image.open(buf).convert('RGB'))
        h, w, _ = image_arr.shape
        image = self.transforms(image=image_arr)['image']
        h_pad, w_pad = max((w - h)//2, 0), max((h - w)//2, 0)
        image_arr = cv2.copyMakeBorder(
            image_arr, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT, 0.)
        encoding = self.processor(
            images=image,
            random_padding=True,
            add_special_tokens=True,
            max_patches=self.cfg.max_patches
        )
        encoding = {k: torch.from_numpy(v[0]) for k, v in encoding.items()}

        # label: ['source', 'chart-type', 'plot-bb', 'text', 'axes', 'data-series', 'id', 'key_point']
        json_dict = json.loads(label)

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

    batch = {"flattened_patches": [], "attention_mask": []}
    texts = [item['text'] for item in samples]

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

    for item in samples:
        batch["flattened_patches"].append(item["flattened_patches"])
        batch["attention_mask"].append(item["attention_mask"])
    batch["flattened_patches"] = torch.stack(batch["flattened_patches"])
    batch["attention_mask"] = torch.stack(batch["attention_mask"])

    batch["id"] = [x["id"] for x in samples]
    if phase == 'valid':
        batch['info'] = [x['info'] for x in samples]
    return batch


def get_transforms(cfg, phase='train'):
    if phase == 'train':
        return A.Compose([
            A.Resize(height=cfg.img_h, width=cfg.img_w),
            A.Normalize(cfg.img_mean, cfg.img_std),
            ToTensorV2()
        ])
    elif phase == 'valid':
        return A.Compose([
            A.Resize(height=cfg.img_h, width=cfg.img_w),
            A.Normalize(cfg.img_mean, cfg.img_std),
            ToTensorV2()
        ])

# Dataloader


def prepare_dataloader(cfg, lmdb_dir, processor, valid_indices):
    valid_ds = MgaDataset(cfg, lmdb_dir, valid_indices,
                          processor, get_transforms(cfg, 'valid'), 'valid')

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.valid_bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return valid_loader


def valid_function(
    cfg,
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

    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
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
            max_length=cfg.max_length,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=10,
            top_k=10,
            top_p=0.9,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True
        )

        outputs.extend(processor.tokenizer.batch_decode(output[0].sequences))
        ids.extend(batch['id'])
        for info in batch['info']:
            table_info_list.append(info)

    scores, pred_list = validation_metrics(outputs, ids, gt_df)
    # create_wandb_table(table_info_list, pred_list, scores)
    return scores


def create_wandb_table(
    table_info_list: List[Dict[str, Any]],
    pred_list: List[Dict[str, Any]],
    scores: Dict[str, Any]
):
    """
    Args:
        table_info_list (List[Dict[str, Any]]):
            dict keys: [img, img_h, img_w, source, x_tick_type, y_tick_type, gt, chart_type]
        pred_list (List[Dict[str, Any]]):
            dict keys: [id, x, y, chart_type, score]
        scores (Dict[str, Any]):
            keys: [valid_score, {chart-type}_score]
    """
    global n_images
    wandb_columns = ['id', 'img', 'gt_x', 'gt_y', 'gt_x_len', 'gt_y_len', 'gt_chart_type', 'pred_x', 'pred_y', 'pred_x_len', 'pred_y_len', 'pred_chart_type', 'score',
                     'n_images', 'img_h', 'img_w', 'source', 'x_tick_type', 'y_tick_type', 'valid_score']
    wandb_data = []

    for pred_dict, info_dict in zip(pred_list, table_info_list):
        data_list = [
            pred_dict['id'],  # id
            wandb.Image(info_dict['img']),  # img
            info_dict['gt_x'],  # gt_x
            info_dict['gt_y'],  # gt_y
            len(info_dict['gt_x']),  # gt_x_len
            len(info_dict['gt_y']),  # gt_y_len
            info_dict['chart_type'],  # gt_chart_type
            pred_dict['x'],  # pred_x
            pred_dict['y'],  # pred_y
            len(pred_dict['x']),  # pred_x_len
            len(pred_dict['y']),  # pred_y_len
            pred_dict['chart_type'],  # pred_chart_type
            pred_dict['score'],
            n_images,  # n_images
            info_dict['img_h'],  # img_h
            info_dict['img_w'],  # img_w
            info_dict['source'],  # source
            info_dict['x_tick_type'],  # x_tick_type
            info_dict['y_tick_type'],  # y_tick_type
            scores['valid_score']  # valid_score
        ]
        wandb_data.append(data_list)

    wandb_table = wandb.Table(columns=wandb_columns, data=wandb_data)
    wandb.log({'valid result': wandb_table})

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
                       name=f'{exp_name.replace("exp", "valid")}', config=wandb.config)
            wandb.config.fold = fold

        # restart or load pretrained model from internet
        pretrained_path = SAVE_DIR.parent / exp_name

        # TODO: save dirにrestartで取ってこれるようにepochやbest scoreをjsonで保存するように実装
        # config
        global max_length, processor, new_tokens, pad_token_id, n_images, best_score
        max_length = cfg.max_length
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

        # learning info
        with open(pretrained_path / 'best_score_info.json', 'r') as f:
            best_score_dict = json.load(f)
            n_images = best_score_dict[str(fold)]['n_images']

        print(f'load model: {str(pretrained_path)}')

        # data indices
        valid_indices = indices_per_fold[fold]['valid']

        # dataloader
        valid_loader = prepare_dataloader(
            cfg, LMDB_DIR, processor, valid_indices)

        valid_score = valid_function(
            cfg, valid_loader, processor, model, device, indices_per_fold[fold]['gt_df'])
        print(f'    VALID:')
        for valid_score_name, valid_score_value in valid_score.items():
            print(
                f'            {valid_score_name}: {valid_score_value:.6f}')
    wandb.finish()
    del model, processor, valid_loader, valid_indices


if __name__ == '__main__':
    main()
