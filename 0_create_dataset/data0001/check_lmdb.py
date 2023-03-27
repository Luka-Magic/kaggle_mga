import json
from tqdm import tqdm
from pathlib import Path
import lmdb
import six
from PIL import Image
from pprint import pprint

def main():
    ROOT_DIR = Path.cwd().parents[2]
    EXP_NAME = Path.cwd().stem
    RAW_LMDB_DIR = ROOT_DIR / 'data' / 'data0001' / 'lmdb'
    SAVE_LMDB_DIR = ROOT_DIR / 'data' / EXP_NAME / 'lmdb'

    raw_env = lmdb.open(str(RAW_LMDB_DIR), max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)


    with raw_env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))

    save_env = lmdb.open(str(SAVE_LMDB_DIR), map_size=int(4e+9))

    count = 0

    for i in range(n_samples):
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
            pass
            


if __name__ == '__main__':
    main()