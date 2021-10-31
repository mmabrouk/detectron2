import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import os

def main(opt):
    os.makedirs(opt.save_dir, exist_ok=True)
    with open(opt.pkl_file, "rb") as f:
        datum = pickle.load(f)
    for data in tqdm(datum):
        im_name = os.path.splitext(os.path.basename(data["file_name"]))[0]
        try:
            im = data["pred_densepose"][0].labels
            im = im.numpy().astype(np.uint8)
            np.save(os.path.join(opt.save_dir, im_name + ".npy"), im)
        except Exception as e:
            print(e)
            print(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_file", "-p", help="pickle file path")
    parser.add_argument("--save_dir", "-s", help="save path")
    opt = parser.parse_args()
    main(opt)