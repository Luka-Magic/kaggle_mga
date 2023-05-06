import json
from tqdm import tqdm
from pathlib import Path
import lmdb
import re
import cv2
import numpy as np

# def anns2point_coors(json_dict, img_size):
#     tick_id2text = {text_dict['id']: text_dict['text'] for text_dict in json_dict['text']}

#     axis_dict = json_dict['axes']

#     tick2coor = {}

#     # horizontalの場合は特殊
#     if json_dict['chart-type'] == 'horizontal_bar':
#         x_ticks_temp = axis_dict['x-axis']['ticks']
#         axis_dict['x-axis']['ticks'] = axis_dict['y-axis']['ticks']
#         axis_dict['y-axis']['ticks'] = x_ticks_temp
#     # dotはxがcategoricalっぽいのにnumericalとか言ってる
#     if json_dict['chart-type'] == 'dot':
#         json_dict['axes']['x-axis']['values-type'] = 'categorical'

#     def _fix_text(text):
#         text = str(text)
#         text = re.sub('\.$', '', text)
#         text = text.replace(',', '').replace('%', '').replace('..', '.').replace('$', '')
#         return float(text)

#     for ax_key, ax_dict in axis_dict.items():
#         ax_type = ax_key.split('-')[0] # x or y
#         if ax_dict['values-type'] == 'numerical':
#             coors, values = [], []
#             for tick_dict in ax_dict['ticks']:
#                 tick_id = tick_dict['id']
#                 text = tick_id2text[tick_id]
#                 coor = tick_dict['tick_pt'][ax_type]

#                 values.append(_fix_text(text))
#                 coors.append(coor)
#             a = (coors[-1] - coors[0]) / (values[-1] - values[0])
#             b = coors[-1] - a * values[-1]
#             tick2coor[ax_type] = {'values-type': 'numerical', 'convert': (a, b)}

#         elif ax_dict['values-type'] == 'categorical':
#             text2coor = {}
#             for tick_dict in ax_dict['ticks']:
#                 tick_id = tick_dict['id']
#                 text = tick_id2text[tick_id]
#                 coor = tick_dict['tick_pt'][ax_type]
#                 text2coor[text] = coor
#             tick2coor[ax_type] = {'values-type': 'categorical', 'convert': text2coor}

#     ann_coors = []
#     for ann in json_dict['data-series']:
#         ann_coor = {}
#         data = {'x': ann['x'], 'y':ann['y']}
#         for ax_type, ax_tick2coor in tick2coor.items():
#             try:
#                 if ax_tick2coor['values-type'] == 'numerical':
#                     a, b = ax_tick2coor['convert']
#                     ann_coor[ax_type] = a * data[ax_type] + b
#                 else:
#                     text2coor = ax_tick2coor['convert']
#                     ann_coor[ax_type] = text2coor[data[ax_type]]
#             except:
#                 pass
#         # 条件
#         if 'x' not in ann_coor or 'y' not in ann_coor:
#             continue
#         if np.isnan(ann_coor['x']) or np.isnan(ann_coor['y']):
#             continue
#         if ann_coor['x'] <= 0 or ann_coor['x'] >= img_size[1] \
#             or ann_coor['y'] <= 0 or ann_coor['y'] >= img_size[0]:
#             continue
#         ann_coors.append(ann_coor)
#     return ann_coors


def add_info_json(json_dict, json_path, img_size):
    id_ = json_path.stem
    json_dict['id'] = id_
    json_dict['image-size'] = {'height': img_size[0], 'width': img_size[1]}
    return json_dict


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def main():
    ROOT_DIR = Path.cwd().parents[2]
    EXP_NAME = Path.cwd().stem
    RAW_DATA_DIR = ROOT_DIR / 'original_data'
    TRAIN_IMG_DIR = RAW_DATA_DIR / 'train' / 'images'
    TEST_IMG_DIR = RAW_DATA_DIR / 'train' / 'images'
    TRAIN_LABEL_DIR = RAW_DATA_DIR / 'train' / 'annotations'

    SAVE_DIR = ROOT_DIR / 'data' / EXP_NAME
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    file_size = 3e9

    img_path_list = list(TRAIN_IMG_DIR.glob('*.jpg'))
    LMDB_DIR = SAVE_DIR / 'lmdb'
    LMDB_DIR.mkdir(exist_ok=True, parents=True)

    env = lmdb.open(str(LMDB_DIR), map_size=int(file_size*1.2))
    cache = {}
    n_images = len(img_path_list)

    count = 1

    for i, img_path in tqdm(enumerate(img_path_list), total=n_images):
        id_ = img_path.stem
        # img
        with open(img_path, 'rb') as f:
            img_bin = f.read()
        h, w, _ = cv2.imread(str(img_path)).shape
        img_size = [h, w]
        # json
        json_path = TRAIN_LABEL_DIR / (img_path.stem + '.json')
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
        try:
            json_dict = add_info_json(json_dict, json_path, img_size)
        except:
            continue
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
