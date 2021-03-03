import os
import sys
import cv2
import glob
import h5py
import numpy as np
import argparse
from utils import gather_per_image
import pickle
from tqdm import tqdm


def h36m_process(annot_file, dataset_path, out_path, split='train'):
    if split == 'train':
        out_file = os.path.join(out_path, 'train.pkl')
    elif split == 'val_p1':
        out_file = os.path.join(out_path, 'val.pkl')
    elif split == 'val_p2':
        out_file = os.path.join(out_path, 'val_p2.pkl')

    annot = np.load(annot_file)
    data_dict = {}
    length = len(annot['imgname'])

    if split == 'train':
        data_dict['filename'] = annot['imgname']
        data_dict['kpts2d'] = annot['part']
        data_dict['kpts3d'] = annot['S']
        data_dict['pose'] = annot['pose']
        data_dict['shape'] = annot['shape']
        data_dict['has_smpl'] = np.ones(length)

        # get bbox
        bboxes = np.zeros([length, 4])
        kp2ds = annot['part']
        for i in tqdm(range(length)):
            kp = kp2ds[i]
            valid = kp[:, -1] > 0
            kp = kp[valid, :-1]
            bbox = np.hstack([np.min(kp, axis=0), np.max(kp, axis=0)])
            bboxes[i, :] = bbox
        data_dict['bboxes'] = bboxes
        # self.boxes = np.hstack([np.min(self.kp2ds, axis=1), np.max(self.kp2ds, axis=1)])
    else:
        data_dict['filename'] = annot['imgname']
        # data_dict['kpts2d'] = annot['part']
        data_dict['kpts3d'] = annot['S']
        # data_dict['pose'] = annot['pose']
        # data_dict['shape'] = annot['shape']
        # data_dict['has_smpl'] = np.ones(length)

        # get bbox
        bboxes = np.zeros([length, 4])
        scale = annot['scale']
        center = annot['center']
        bboxes[:, :2] = center - 200 * scale[:, None] * 0.5
        bboxes[:, 2:] = center + 200 * scale[:, None] * 0.5
        data_dict['bboxes'] = bboxes
        # self.boxes = np.hstack([np.min(self.kp2ds, axis=1), np.max(self.kp2ds, axis=1)])



    data = gather_per_image(data_dict, img_dir=dataset_path)
    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)



# Illustrative script for training data extraction
# No SMPL parameters will be included in the .npz file.
def h36m_extract(dataset_path, out_path, split='train'):

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, widths_, heights_, centers_, parts_, Ss_  = [], [], [], [], [], []

    if split == 'train':
        user_list = [1, 5, 6, 7, 8]
        out_file = os.path.join(out_path, 'train.pkl')
        protocol = 1
    elif split == 'val_p1':
        user_list = [9, 11]
        out_file = os.path.join(out_path, 'val.pkl')
        protocol = 1
    elif split == 'val_p2':
        user_list = [9, 11]
        out_file = os.path.join(out_path, 'val_p2.pkl')
        protocol = 2

    annotations = []
    # go over each user
    for user_i in user_list:
        user_name = 'S%d' % user_i
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat', 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D2_Positions')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        for seq_i in seq_list:

            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            action = action.replace(' ', '_')
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            poses_3d = pycdf.CDF(seq_i)['Pose'][0]

            # 2D pose file
            pose2d_file = os.path.join(pose2d_path, seq_name)
            poses_2d = pycdf.CDF(pose2d_file)['Pose'][0]

            # bbox file
            bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            bbox_h5py = h5py.File(bbox_file)

            # video file
            vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
            imgs_path = os.path.join(dataset_path, 'images')
            vidcap = cv2.VideoCapture(vid_file)

            # go over each frame of the sequence
            for frame_i in range(poses_3d.shape[0]):
                # read video frame
                success, image = vidcap.read()
                if not success:
                    break

                # check if you can keep this frame
                if frame_i % 5 == 0 and (protocol == 1 or camera == '60457274'):
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)

                    # save image
                    img_out = os.path.join(imgs_path, imgname)
                    h, w, _ = img_out.shape
                    cv2.imwrite(img_out, image)

                    # read GT bounding box
                    mask = bbox_h5py[bbox_h5py['Masks'][frame_i,0]].value.T
                    ys, xs = np.where(mask==1)
                    bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])

                    # read GT 3D pose
                    partall = np.reshape(poses_2d[frame_i,:], [-1,2])
                    part17 = partalll[h36m_idx]
                    part = np.zeros([24,3])
                    part[global_idx, :2] = part17
                    part[global_idx, 2] = 1

                    # read GT 3D pose
                    Sall = np.reshape(poses_3d[frame_i,:], [-1,3])/1000.
                    S17 = Sall[h36m_idx]
                    S17 -= S17[0] # root-centered
                    S24 = np.zeros([24,4])
                    S24[global_idx, :3] = S17
                    S24[global_idx, 3] = 1

                    datum = dict(filename=os.path.join('images', imgname),
                                 width=w,
                                 height=h,
                                 bboxes=np.array(bbox)[np.newaxis],
                                 kpts2d=part[np.newaxis],
                                 kpts3d=S24[np.newaxis])
                    annotations.append(datum)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    # h36m_extract(args.dataset_path, args.out_path, split=args.split)
    train_path = '/home/wzeng/mydata/H36Mnew/c2f_vol'
    val_path = '/home/wzeng/mydata/MyH36MOrigin'
    out_path = '/home/wzeng/mydata/H36Mnew/c2f_vol/rcnn'

    # h36m_process('/home/wzeng/mydata/DecoMR/h36m_train_new.npz', train_path, out_path, split='train')
    h36m_process('/home/wzeng/mydata/DecoMR/h36m_valid_protocol1.npz', val_path, out_path, split='val_p1')
    h36m_process('/home/wzeng/mydata/DecoMR/h36m_valid_protocol2.npz', val_path, out_path, split='val_p2')

