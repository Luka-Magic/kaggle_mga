import json
from tqdm import tqdm
from pathlib import Path
import lmdb
import six
from PIL import Image
from pprint import pprint
import sys


def main():
    phase = sys.argv[1]
    ROOT_DIR = Path.cwd().parents[2]
    EXP_NAME = Path.cwd().stem
    LMDB_DIR = ROOT_DIR / 'data' / EXP_NAME / phase / 'lmdb'

    env = lmdb.open(str(LMDB_DIR), max_readers=32, readonly=True,
                    lock=False, readahead=False, meminit=False)

    with env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))
        print(n_samples)
        for i in [1, n_samples-1]:
            i += 1

            # image
            img_key = f'image-{str(i).zfill(8)}'.encode()
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            print(img.size)

            # json
            label_key = f'label-{str(i).zfill(8)}'.encode()
            label = txn.get(label_key).decode('utf-8')
            json_dict = json.loads(label)
            pprint(json_dict)


if __name__ == '__main__':
    main()
