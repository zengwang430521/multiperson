import os
import argparse
import cv2
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser(description='Preprocess MPI-INF-3DHP')
parser.add_argument('--dataset_path')
parser.add_argument('--ann_file')


def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i*7+5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i*7+6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3,:3]
        T = RT[:3,3]/1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts


def extract_img(dataset_path, ann_file):
    h, w = 2048, 2048

    # training data
    user_list = range(1,9)
    seq_list = range(1,3)
    vid_list = list(range(3)) + list(range(4,9))
    with open(ann_file, 'rb') as f:
        raw_infos = pickle.load(f)
    save_imgnames = [item['filename'] for item in raw_infos]

    for user_i in tqdm(user_list):
        for seq_i in tqdm(seq_list):
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i))
            # mat file with annotations
            annot_file = os.path.join(seq_path, 'annot.mat')
            annot2 = sio.loadmat(annot_file)['annot2']
            annot3 = sio.loadmat(annot_file)['annot3']
            # calibration file and camera parameters
            calib_file = os.path.join(seq_path, 'camera.calibration')
            Ks, Rs, Ts = read_calibration(calib_file, vid_list)

            for j, vid_i in enumerate(vid_list):

                # image folder
                imgs_path = os.path.join(seq_path,
                                         'imageFrames',
                                         'video_' + str(vid_i))

                # extract frames from video file
                # if doesn't exist
                if not os.path.isdir(imgs_path):
                    os.makedirs(imgs_path)

                # video file
                vid_file = os.path.join(seq_path,
                                        'imageSequence',
                                        'video_' + str(vid_i) + '.avi')
                vidcap = cv2.VideoCapture(vid_file)

                # process video
                frame = 0
                while 1:
                    # extract all frames
                    success, image = vidcap.read()
                    if not success:
                        break
                    frame += 1
                    # image name
                    imgname = os.path.join(imgs_path, 'frame_%06d.jpg' % frame)
                    part_name = os.path.join('S' + str(user_i),
                                            'Seq' + str(seq_i),
                                            'imageFrames',
                                            'video_' + str(vid_i),
                                            'frame_%06d.jpg' % frame)
                    # save image
                    if part_name in save_imgnames:
                        cv2.imwrite(imgname, image)


if __name__ == '__main__':
    args = parser.parse_args()
    extract_img(args.dataset_path, args.ann_file)
    print('finish')
