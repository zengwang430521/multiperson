import argparse
from utils import gather_per_image
import os
import numpy as np
import scipy.io as sio
import pickle

parser = argparse.ArgumentParser(description='Preprocess LSPET')
parser.add_argument('dataset_path')
parser.add_argument('out_path')
import os
from os.path import join
import argparse
import numpy as np
import scipy.io as sio
import cv2
import pickle
import glob
from tqdm import trange

def lspet_extract(dataset_path, out_path, out_size=256):

    imgnames_, bboxes_, kpts2d_ = [], [], []

    png_path = os.path.join(os.path.join(dataset_path, 'images'), '*.jpg')
    imgs = glob.glob(png_path)
    imgs.sort()

    # annotation files
    annot_file = os.path.join(dataset_path, 'joints.mat')
    joints = sio.loadmat(annot_file)['joints'].astype(np.float)

    # go over all the images
    for i in trange(len(imgs)):
        # image name
        imgname = imgs[i].split('/')[-1]
        part14 = joints[:, :2, i]
        bbox = [min(part14[:,0]), min(part14[:,1]),
                max(part14[:,0]), max(part14[:,1])]

        part = np.zeros([24,3])
        part[:14] = np.hstack([part14, np.ones([14,1])])


        imgnames_.append(imgname)
        bboxes_.append(bbox)
        kpts2d_.append(part)

    imgnames = np.array(imgnames_)
    bboxes = np.array(bboxes_)
    kpts2d = np.array(kpts2d_)

    data = gather_per_image(dict(filename=imgnames, bboxes=bboxes, kpts2d=kpts2d),
                            img_dir=os.path.join(dataset_path, 'images'))

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'train.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    args = parser.parse_args()
    lspet_extract(args.dataset_path, args.out_path)
