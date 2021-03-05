import argparse
from utils import gather_per_image
import os
import numpy as np
import scipy.io as sio
import pickle

parser = argparse.ArgumentParser(description='Preprocess LSP')
parser.add_argument('dataset_path')
parser.add_argument('out_path')


def lsp_extract(dataset_path, out_path):

    # structs we use
    imgnames_, bboxes_, kpts2d_ = [], [], []

    # we use LSP dataset original for training
    imgs = range(1000)

    # annotation files
    annot_file = os.path.join(dataset_path, 'joints.mat')
    joints = sio.loadmat(annot_file)['joints']

    # go over all the images
    for img_i in imgs:
        # image name
        imgname = 'im%04d.jpg' % (img_i+1)
        # read keypoints
        part14 = joints[:2,:,img_i].T
        # scale and center
        bbox = [min(part14[:,0]), min(part14[:,1]),
                max(part14[:,0]), max(part14[:,1])]
        # update keypoints
        part = np.zeros([24,3])
        part[:14] = np.hstack([part14, np.ones([14,1])])

        imgnames_.append(imgname)
        bboxes_.append(bbox)
        kpts2d_.append(part)

    imgnames = np.array(imgnames_)
    bboxes = np.array(bboxes_)
    kpts2d = np.array(kpts2d_)

    data = gather_per_image(dict(filename=imgnames, bboxes=bboxes, kpts2d=kpts2d), img_dir=os.path.join(dataset_path, 'images'))

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'train.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    args = parser.parse_args()
    lsp_extract(args.dataset_path, args.out_path)
