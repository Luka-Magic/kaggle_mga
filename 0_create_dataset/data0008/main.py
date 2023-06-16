import json
from tqdm import tqdm
from pathlib import Path
import lmdb
import re
import cv2
import numpy as np
import pandas as pd
from operator import itemgetter


def check_ann(ann, chart_type):
    if 'task6' not in ann or ann['task6'] is None:
        return False
    if chart_type != 'scatter':
        if len(ann['task6']['output']['data series']) != 1:
            return False
    return True


def ann2json(ann, id_, chart_type, img_path):
    '''
    json
        id
        source
        data-series
        count
        chart-type
    '''
    json_dict = {}
    json_dict['id'] = id_
    h, w, _ = cv2.imread(str(img_path)).shape
    json_dict['image-size'] = {'height': h, 'width': w}

    raw_data_series = ann['task6']['output']['data series']
    data_series = []
    count = 0
    for raw_data in raw_data_series:
        for data_dict in raw_data['data']:
            if 'x' not in data_dict or 'y' not in data_dict:
                continue
            data_series.append({'x': data_dict['x'], 'y': data_dict['y']})
            count += 1

    if chart_type == 'scatter':
        json_dict['data-series'] = sorted(data_series,
                                          key=itemgetter('x', 'y'))
    else:
        json_dict['data-series'] = data_series
    json_dict['chart-type'] = chart_type
    json_dict['count'] = count
    json_dict['source'] = 'icdar2022'
    json_dict['visual-elements'] = ann['task6']['output']['visual elements']
    return json_dict


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def main():
    ROOT_DIR = Path.cwd().parents[2]
    EXP_NAME = Path.cwd().stem
    RAW_DATA_DIR = ROOT_DIR / 'extra_data/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0'
    IMAEG_DATA_DIR = RAW_DATA_DIR / 'images'
    TRAIN_LABEL_PATH = RAW_DATA_DIR / 'annotations_JSON'
    CHART_TYPES = [
        'scatter',
        'line',
        'vertical_bar',
        'horizontal_bar'
    ]
    count = 1

    SAVE_DIR = ROOT_DIR / 'data' / EXP_NAME
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    file_size = 3e10

    LMDB_DIR = SAVE_DIR / 'lmdb'
    LMDB_DIR.mkdir(exist_ok=True, parents=True)

    env = lmdb.open(str(LMDB_DIR), map_size=int(file_size*1.2))
    cache = {}

    img_paths = list(IMAEG_DATA_DIR.glob('*/*.jpg'))
    for i, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
        id_ = img_path.stem
        chart_type = img_path.parent.stem
        ann_path = TRAIN_LABEL_PATH / chart_type / f'{id_}.json'

        # 使うchart_typeのみ
        if chart_type not in CHART_TYPES:
            continue

        # json
        with open(str(ann_path), 'r') as f:
            ann = json.load(f)

        # check data
        can_use_data = check_ann(ann, chart_type)
        if not can_use_data:
            continue

        # img
        with open(str(img_path), 'rb') as f:
            img_bin = f.read()

        # json_dict
        json_dict = ann2json(ann, id_, chart_type, img_path)
        json_bin = json.dumps(json_dict).encode()

        # key
        img_key = f'image-{str(count).zfill(8)}'.encode()
        label_key = f'label-{str(count).zfill(8)}'.encode()

        cache[img_key] = img_bin
        cache[label_key] = json_bin

        if i % 1000 == 0:
            cache['num-samples'.encode()] = str(count).encode()
            write_cache(env, cache)
            cache = {}

        count += 1
    cache['num-samples'.encode()] = str(count - 1).encode()
    write_cache(env, cache)


if __name__ == '__main__':
    main()
