import numpy as np
from pathlib import Path
import lmdb
import six
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import japanize_matplotlib
import json

def get_charactor(lmdb_dir):
    env = lmdb.open(str(lmdb_dir), max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

    text_set = set()
    c = {}

    with env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))
        for i in tqdm(range(n_samples), total=n_samples):
            i += 1
            # # image
            # img_key = f'image-{str(i).zfill(8)}'.encode()
            # imgbuf = txn.get(img_key)
            # buf = six.BytesIO()
            # buf.write(imgbuf)
            # buf.seek(0)
            # img = Image.open(buf).convert('RGB')

            # json
            label_key = f'label-{str(i).zfill(8)}'.encode()
            label = txn.get(label_key).decode('utf-8')
            json_dict = json.loads(label)
            
            for s in json_dict['text']:
                # if s not in c:
                #     c[s] = 0
                # c[s] += 1
                text_set.add(s)
            
            # one_text_set = set([s for s in json_dict['text']])
            # text_set += one_text_set
    charactors = list(text_set)
    charactors.sort()
    charactors = ''.join(charactors)
    return charactors

def main():
    ROOT_DIR = Path.cwd().parents[2]
    EXP_NAME = Path.cwd().stem
    LMDB_DIR = ROOT_DIR / 'data' / EXP_NAME / 'lmdb'
    CHAR_PATH = ROOT_DIR / 'data' / EXP_NAME / 'charactor.txt'

    charactors = get_charactor(LMDB_DIR)

    with open(str(CHAR_PATH), 'w') as f:
        f.write(charactors)

if __name__ == '__main__':
    main()
