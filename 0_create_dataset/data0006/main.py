import json
from tqdm import tqdm
from pathlib import Path
import lmdb
import re
import cv2
import numpy as np
import pandas as pd


def add_info_json(json_dict, json_path, img_size):
    id_ = json_path.stem
    json_dict['id'] = id_
    json_dict['image-size'] = {'height': img_size[0], 'width': img_size[1]}
    return json_dict


def text2data_series(text):
    data_list = text.split(' <0x0A> ')
    xs = []
    ys = []
    for one_data in data_list:
        assert ' | ' in one_data, f'"{one_data}" has no signal of " | "'
        one_data = one_data.split(' | ')
        assert len(one_data) == 2, f'length of list "{one_data}" is not 2'
        xs.append(one_data[0])
        ys.append(one_data[1])
    data_series = [{'x': x, 'y': y} for x, y in zip(xs, ys)]
    return data_series


def gt2json(row, img_path):
    '''
    json
        id
        source
        data-series

    '''
    json_dict = {}
    json_dict['id'] = row['file_name'].replace('/', '_').replace('.jpg', '')
    h, w, _ = cv2.imread(str(img_path)).shape
    json_dict['image-size'] = {'height': h, 'width': w}

    text = row['text']
    json_dict['data-series'] = text2data_series(text)
    json_dict['chart-type'] = row['chart_type']
    json_dict['count'] = row['count']
    json_dict['source'] = 'bartley_1'
    return json_dict


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def main():
    ROOT_DIR = Path.cwd().parents[2]
    EXP_NAME = Path.cwd().stem
    RAW_DATA_DIR = ROOT_DIR / 'kaggle_dataset/benetech-extra-generated-data'
    IMAEG_DATA_DIR = RAW_DATA_DIR
    TRAIN_LABEL_PATH = RAW_DATA_DIR / 'metadata.csv'

    df = pd.read_csv(TRAIN_LABEL_PATH)
    df_dict = {
        'train': df.query('validation==0'),
        'valid': df.query('validation==1')
    }

    for phase in ['train', 'valid']:
        count = 1
        phase_df = df_dict[phase]

        SAVE_DIR = ROOT_DIR / 'data' / EXP_NAME / phase
        SAVE_DIR.mkdir(exist_ok=True, parents=True)
        file_size = 3e10

        LMDB_DIR = SAVE_DIR / 'lmdb'
        LMDB_DIR.mkdir(exist_ok=True, parents=True)

        env = lmdb.open(str(LMDB_DIR), map_size=int(file_size*1.2))
        cache = {}

        for i, row in tqdm(phase_df.iterrows(), total=len(phase_df)):
            img_path = IMAEG_DATA_DIR / row['file_name']
            id_ = row['file_name'].replace('/', '_').replace('.jpg', '')

            # img
            with open(img_path, 'rb') as f:
                img_bin = f.read()

            # json
            json_dict = gt2json(row, img_path)
            # try:
            #     json_dict = gt2json(row, img_path)
            # except:
            #     continue
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
        print(f'{phase}: count{count}')
        write_cache(env, cache)


if __name__ == '__main__':
    main()
