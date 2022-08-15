# modified from: https://github.com/microsoft/voxelpose-pytorch

import glob
import os.path as osp
import argparse
import numpy as np
import json
import os
from tqdm import tqdm
from copy import deepcopy


TRAIN_LIST = [
    '160224_haggling1',
    '160226_mafia2',
    '160224_mafia1',
    '160224_mafia2',
    '160224_ultimatum1',
    '160224_ultimatum2',
]

VAL_LIST = [
    '160422_mafia2',  # 15
    '160226_haggling1',  # 19
    '160422_haggling1',  # 19
    '160226_mafia1',  # 15
    '160226_ultimatum1',  # TODO: check annotations
    '160422_ultimatum1',  # 19
    '160906_pizza1',  # 19
]

JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
    # 'l-eye': 15,
    # 'l-ear': 16,
    # 'r-eye': 17,
    # 'r-ear': 18,
}

LIMBS = [
    [0, 1],
    [0, 2],
    [0, 3],
    [3, 4],
    [4, 5],
    [0, 9],
    [9, 10],
    [10, 11],
    [2, 6],
    [2, 12],
    [6, 7],
    [7, 8],
    [12, 13],
    [13, 14]]

DATA_ROOT = ''


def projectPoints(X, K, R, t, Kd):
    """
    Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
    Roughly, x = K*(R*X + t) + distortion
    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """

    x = np.dot(R, X) + t

    x[0:2, :] = x[0:2, :] / (x[2, :] + 1e-5)

    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

    x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                         ) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
                      r + 2 * x[0, :] * x[0, :])
    x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                         ) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
                      r + 2 * x[1, :] * x[1, :])

    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    return x


class Panoptic():
    def __init__(self, image_set=None):
        self.pixel_std = 200.0
        self.joints_def = JOINTS_DEF
        self.root_id = 2
        self.limbs = LIMBS
        self.num_joints = len(JOINTS_DEF)
        self.dataset_root = DATA_ROOT
        self.cam_list = [(0, 16), (0, 30)]
        self.num_views = len(self.cam_list)

        # train set
        if image_set == 'train':
            self.sequence_list = TRAIN_LIST
            self._interval = 2
            self.train_mode = True
        # validation set
        elif 'haggling' in image_set:
            self.sequence_list = [
                '160226_haggling1',
                '160422_haggling1'
            ]
            self._interval = None
            self._total = 2400
        elif 'mafia' in image_set:
            self.sequence_list = [
                '160226_mafia1',
                '160422_mafia2'
            ]
            self._interval = None
            self._total = 2400
        elif 'ultimatum' in image_set:
            self.sequence_list = [
                # '160226_ultimatum1',
                '160422_ultimatum1'
            ]
            self._interval = None
            self._total = 2400
        elif 'pizza' in image_set:
            self.sequence_list = [
                '160906_pizza1',
            ]
            self._interval = None
            self._total = 2400

        self.json_file = 'annotations/{}.json'.format(image_set)
        self.json_file = os.path.join(self.dataset_root, self.json_file)
        os.makedirs(os.path.dirname(self.json_file), exist_ok=True)

        self.db = self._get_db()
        self.db_size = len(self.db['images'])
        print('db size:', self.db_size)

        self.write_db()

    def _get_db(self):
        width = 1920
        height = 1080
        image_info = []
        anno_info = []
        category_info = [{
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": list(sorted(JOINTS_DEF.keys(), key=lambda x: JOINTS_DEF[x])),
            "skeleton": LIMBS,
        }]
        img_count = 1
        anno_count = 1
        prev_ind = 0

        for seq in self.sequence_list:
            cameras = self._get_cam(seq)

            _interval = self._interval
            if hasattr(self, '_total') and self._total:
                _total = self._total // len(self.sequence_list) // len(cameras)
                assert _interval is None
            else:
                _total = None
                assert _interval is not None
            _train_mode = hasattr(self, 'train_mode') and self.train_mode

            # load annotations
            curr_anno = osp.join(self.dataset_root, seq, 'hdPose3d_stage1_coco19')
            anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))
            joints_key = 'joints19'
            if len(anno_files) == 0:
                curr_anno = osp.join(self.dataset_root, seq, 'hdPose3d_stage1')
                anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))
                joints_key = 'joints15'

            print(seq)
            for k, v in cameras.items():
                for i, file in tqdm(list(enumerate(anno_files))):
                    if _interval and i % _interval == 0 or _total:
                        with open(file) as dfile:
                            try:
                                bodies = json.load(dfile)['bodies']
                            except Exception as e:
                                print(e)
                                continue
                        if len(bodies) == 0:
                            continue

                        postfix = osp.basename(file).replace('body3DScene', '')
                        prefix = '{:02d}_{:02d}'.format(k[0], k[1])
                        image = osp.join(seq, 'hdImgs', prefix,
                                         prefix + postfix)
                        image = image.replace('json', 'jpg')
                        if not os.path.exists(os.path.join(self.dataset_root, image)):
                            print('WARNING: File Not Exist', os.path.join(self.dataset_root, image))
                            continue

                        tmp_img_count = 0
                        tmp_anno_count = 0

                        img_inst = {
                            "id": img_count + tmp_img_count,
                            "width": width,
                            "height": height,
                            "file_name": image,
                        }
                        tmp_img_count += 1
                        anno_insts = []
                        invalid_count = 0

                        all_poses_3d = []
                        all_poses_vis_3d = []
                        all_poses = []
                        all_poses_vis = []
                        for body in bodies:
                            pose3d = np.array(body[joints_key]).reshape((-1, 4))
                            pose3d = pose3d[:self.num_joints]

                            joints_vis = pose3d[:, -1] > 0.1

                            # MPII annotation format
                            if joints_key == 'joints19':
                                joints_vis[1] = 0

                            M = np.array([[1.0, 0.0, 0.0],
                                          [0.0, 0.0, -1.0],
                                          [0.0, 1.0, 0.0]])
                            pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)

                            all_poses_3d.append(pose3d[:, 0:3] * 10.0)
                            all_poses_vis_3d.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 3, axis=1))
                            joints_for_bbox = joints_vis.copy()

                            pose2d = np.zeros((pose3d.shape[0], 2))
                            pose_img = projectPoints(
                                pose3d[:, 0:3].transpose(), v['K'], v['R'],
                                v['t'], v['distCoef']).transpose()
                            pose2d[:, :2] = pose_img[:, :2]
                            x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                                     pose2d[:, 0] <= width - 1)
                            y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                     pose2d[:, 1] <= height - 1)
                            check = np.bitwise_and(x_check, y_check)
                            joints_vis[np.logical_not(check)] = 0

                            all_poses.append(pose2d)
                            all_poses_vis.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 2, axis=1))

                            center2d = pose_img[self.root_id]
                            if joints_for_bbox.sum() < 3:
                                invalid_count += 1
                                continue

                            # bounding box generation
                            xmin, ymin = np.min(pose2d[joints_for_bbox], axis=0)
                            xmax, ymax = np.max(pose2d[joints_for_bbox], axis=0)
                            w, h = xmax - xmin, ymax - ymin
                            if joints_key == 'joints19':
                                ymin, ymax = ymin - 0.30 * h, ymax + 0.15 * h
                            elif joints_key == 'joints15':
                                ymin, ymax = ymin - 0.02 * h, ymax + 0.07 * h
                            xmin, xmax = xmin - 0.15 * w, xmax + 0.15 * w
                            xmin, xmax = np.array([xmin, xmax]).clip(0, width - 1)
                            ymin, ymax = np.array([ymin, ymax]).clip(0, height - 1)
                            xmin, xmax, ymin, ymax = np.array([xmin, xmax, ymin, ymax]).tolist()
                            w, h = xmax - xmin + 1, ymax - ymin + 1
                            if w <= 1 or h <= 1 or w * h <= 64:
                                invalid_count += 1
                                continue

                            anno_inst = {
                                "id": anno_count + tmp_anno_count,
                                "image_id": img_inst['id'],
                                "category_id": 1,
                                "area": w * h,
                                "bbox": [xmin, ymin, w, h],
                                "iscrowd": 0,
                                "joints2d": all_poses[-1],
                                "joints2d_vis": all_poses_vis[-1],
                                "joints3d_img": pose_img,
                                "joints3d": all_poses_3d[-1],
                                "joints3d_vis": all_poses_vis_3d[-1],
                                "center2d": center2d,
                                "num_keypoints": joints_vis.sum(),
                            }

                            n_anno_inst = deepcopy(anno_inst)
                            for key, value in anno_inst.items():
                                if isinstance(value, np.ndarray):
                                    if value.dtype == bool:
                                        value = value.astype(np.int32)
                                    if value.dtype == np.int64:
                                        value = value.astype(np.int32)
                                    n_anno_inst[key] = value.tolist()
                                if isinstance(value, np.integer):
                                    n_anno_inst[key] = int(value)
                            anno_insts.append(n_anno_inst)
                            tmp_anno_count += 1

                        if tmp_anno_count > 0 and (not _train_mode or invalid_count == 0):
                            img_inst['cam'] = {
                                'K': v['K'].tolist(),
                                'R': v['R'].tolist(),
                                't': v['t'].tolist(),
                            }
                            image_info.append(img_inst)
                            img_count += tmp_img_count
                            anno_info.extend(anno_insts)
                            anno_count += tmp_anno_count

                if _total:
                    to_check = image_info[prev_ind:]
                    checked = image_info[:prev_ind]
                    assert len(to_check) >= _total
                    sample_inds = np.linspace(0, len(to_check) - 1, _total).astype(int)
                    image_info = checked + [to_check[_] for _ in sample_inds]
                    image_ids = set(x['id'] for x in image_info)
                    anno_info = [x for x in anno_info if x['image_id'] in image_ids]
                    prev_ind += _total

        db = {
            'images': image_info,
            'annotations': anno_info,
            'categories': category_info
        }

        return db

    def _get_cam(self, seq):
        cam_file = osp.join(self.dataset_root, seq, 'calibration_{:s}.json'.format(seq))
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib['cameras']:
            if (cam['panel'], cam['node']) in self.cam_list:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                cameras[(cam['panel'], cam['node'])] = sel_cam
        return cameras

    def write_db(self):
        with open(self.json_file, 'w') as f:
            json.dump(self.db, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/panoptic')
    args = parser.parse_args()
    DATA_ROOT = args.root
    Panoptic('train')
    Panoptic('haggling')
    Panoptic('mafia')
    Panoptic('ultimatum')
    Panoptic('pizza')
