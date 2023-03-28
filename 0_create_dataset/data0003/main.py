import json
from tqdm import tqdm
from pathlib import Path
import lmdb
import re
import six
from PIL import Image, ImageTransform
import io
import numpy as np
import math

def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def main():
    ROOT_DIR = Path.cwd().parents[2]
    EXP_NAME = Path.cwd().stem
    RAW_LMDB_DIR = ROOT_DIR / 'data' / 'data0001' / 'lmdb'
    SAVE_LMDB_DIR = ROOT_DIR / 'data' / EXP_NAME / 'lmdb'
    SAVE_LMDB_DIR.mkdir(parents=True, exist_ok=True)

    raw_env = lmdb.open(str(RAW_LMDB_DIR), max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

    with raw_env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))

    save_env = lmdb.open(str(SAVE_LMDB_DIR), map_size=int(4e+9))

    count = 0
    cache = {}

    for i in tqdm(range(n_samples), total=n_samples):
        i += 1

        with raw_env.begin(write=False) as txn:
            # image
            img_key = f'image-{str(i).zfill(8)}'.encode()
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            
            # json
            label_key = f'label-{str(i).zfill(8)}'.encode()
            label = txn.get(label_key).decode('utf-8')
            json_dict = json.loads(label)

        for text_dict in json_dict['text']:
            if text_dict['role'] != 'tick_label':
                continue

            x0, x1, x2, x3, y0, y1, y2, y3 = text_dict['polygon'].values()

            # image
            # x_min, x_max, y_min, y_max = polygon[:4].min(), polygon[:4].max(), polygon[4:].min(), polygon[4:].max()
            # crop_img = img.copy()
            # crop_img = img.crop((x_min, y_min, x_max, y_max))
            w, h = ((x1-x0)**2 +(y1-y0)**2)**0.5, ((x3-x0)**2 +(y3-y0)**2)**0.5
            crop_img = img.transform((math.floor(w), math.floor(h)), ImageTransform.QuadTransform([x0, y0, x3, y3, x2, y2, x1, y1]))
            crop_img_bytes = io.BytesIO()
            crop_img.save(crop_img_bytes, format='PNG')
            img_bin = crop_img_bytes.getvalue()
            
            # label
            label_dict = {}
            label_dict['id'] = json_dict['id']
            label_dict['text'] = text_dict['text']
            label_dict['polygon'] = text_dict['polygon']
            label_dict['img-size'] = {'height': math.floor(h), 'width': math.floor(w)}
            # label_dict['bbox'] = {'x_min': int(x_min), 'x_max': int(x_max), 'y_min': int(y_min), 'y_max': int(y_max)}
            label_dict['role'] = text_dict['role']
            json_bin = json.dumps(label_dict).encode()

            # key
            save_img_key = f'image-{str(count+1).zfill(8)}'.encode()
            save_label_key = f'label-{str(count+1).zfill(8)}'.encode()

            cache[save_img_key] = img_bin
            cache[save_label_key] = json_bin

            if count % 1000 == 0:
                cache['num-samples'.encode()] = str(count).encode()
                write_cache(save_env, cache)
                cache = {}
            count += 1
        cache['num-samples'.encode()] = str(count).encode()
        write_cache(save_env, cache)

if __name__ == '__main__':
    main()