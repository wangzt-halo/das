# modified from: https://github.com/3dpose/3D-Multi-Person-Pose

import os
import json

import cv2
import numpy as np
import scipy.io as sio
import copy
from collections import OrderedDict, defaultdict
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mytools.vis_3d import pixel2world
from multiprocessing import Process, Manager


@DATASETS.register_module()
class MuPots3DHP(CocoDataset):
    joint_num = 21  # for MuCo-3DHP
    joints_name = (
        'Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee',
        'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe',
        'L_Toe')
    original_joint_num = 17  # for MuPoTS-3D
    original_joints_name = (
        'Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee',
        'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head')  # MuPoTS

    flip_pairs = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13))
    skeleton = (
        (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (11, 12), (12, 13), (1, 2), (2, 3),
        (3, 4), (1, 5), (5, 6), (6, 7))
    eval_joint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

    JOINTS_DEF = {k: i for i, k in enumerate(original_joints_name)}
    ROOT_IDX = joints_name.index('Pelvis')

    def __init__(self, use_bbox_center=False, norm_depth=False, abs_dz=False, depth_factor=1, **kwargs):
        super().__init__(**kwargs)

        self.num_joints = len(self.JOINTS_DEF)
        self.name2id = {}
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            self.name2id[info['file_name']] = i
        self.use_bbox_center = use_bbox_center
        self.norm_depth = norm_depth
        self.depth_factor = depth_factor
        self.abs_dz = abs_dz
        if abs_dz:
            assert norm_depth

    def prepare_train_img(self, idx):
        raise NotImplementedError

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        if ann_info and 'cam' in ann_info:
            n_ann_info = {'cam': ann_info['cam']}
        else:
            n_ann_info = dict()
        n_ann_info['gt_poses_3d'] = ann_info['gt_poses_3d']
        n_ann_info['gt_labels_3d'] = ann_info['gt_labels_3d']
        n_ann_info['centers2d'] = ann_info['centers2d']
        n_ann_info['depths'] = ann_info['depths']
        results = dict(img_info=img_info, ann_info=n_ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def _parse_ann_info(self, img_info, ann_info):
        # generate pseudo camera parameters
        intrinsic = img_info['intrinsic']
        f = intrinsic[:2]
        c = intrinsic[2:]
        cam = dict(
            K=[[f[0], 0., c[0]],
               [0., f[1], c[1]]],
            R=[[1.0, 0.0, 0.0],
               [0.0, 1.0, .0],
               [0.0, 0.0, 1.0]],
            t=[[0.], [0.], [0.]]
        )
        cam = {k: np.array(v) for k, v in cam.items()}

        gt_bboxes = []
        gt_labels = []
        gt_poses_3d = []
        gt_bboxes_ignore = []
        centers2d = []
        depths = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if 'area' in ann and ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                # 3D pose
                pose_img = np.array(ann['keypoints_img'], dtype=np.float)
                pose_cam = np.array(ann['keypoints_cam'], dtype=np.float)
                pose_3d = np.concatenate([pose_img, pose_cam[:, 2:]], axis=1)
                pose_vis = ann['keypoints_vis']
                pose_3d = np.array(pose_3d)
                if self.norm_depth:
                    pose_3d[:, 2] /= self.depth_factor
                    if self.abs_dz:
                        abs_dz = pose_3d[:, 2] - pose_3d[[self.ROOT_IDX], 2]
                    pose_3d[:, 2] /= np.sqrt(f[0]*f[1])
                ann['center2d'] = pose_3d[self.ROOT_IDX].copy()
                if pose_3d.max() - pose_3d.min() < 10:
                    continue
                if not self.use_bbox_center:
                    if pose_vis[self.ROOT_IDX][0] == 0:
                        gt_bboxes_ignore.append(bbox)
                        continue
                    c2d = ann['center2d']
                else:
                    c2d = np.array(ann['center2d']).copy()
                    c2d[0] = x1 + 0.5 * w
                    c2d[1] = y1 + 0.5 * h
                center2d = c2d[:2]
                depth = c2d[2]
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                if self.abs_dz:
                    pose_3d[:, 2] = abs_dz
                general_pose_3d = np.concatenate([
                    np.array(c2d, dtype=np.float).reshape(-1),
                    np.array(pose_3d, dtype=np.float).reshape(-1),
                    np.array(pose_vis, dtype=np.float).reshape(-1)
                ])
                gt_poses_3d.append(general_pose_3d)
                centers2d.append(center2d)
                depths.append(depth)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_poses_3d = np.array(gt_poses_3d, dtype=np.float32)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_poses_3d = np.zeros((0, 3 + self.num_joints * 3 + self.num_joints), dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)
            # return None

        gt_labels_3d = copy.deepcopy(gt_labels)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_poses_3d=gt_poses_3d,
            gt_labels_3d=gt_labels_3d,
            centers2d=centers2d,
            depths=depths,
            bboxes_ignore=gt_bboxes_ignore,
        )
        ann['cam'] = cam

        return ann

    def _filter_imgs(self, min_size=32):
        valid_inds = super(MuPots3DHP, self)._filter_imgs(min_size)
        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            if i in valid_inds:
                img_id = self.img_ids[i]
                filename = img_info['file_name']
                if 'TS%d/' % (self.eval_seq + 1) in filename:
                    valid_inds.append(i)
                    valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def evaluate(self, outputs, res_folder='tmp', metric='pck', eval_mode='all', **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['pck']
        for metric in metrics:
            if metric.lower() not in allowed_metrics:
                raise KeyError(f'metric {metric.lower()} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')

        preds = []
        vis = []
        scores = []
        image_paths = []

        data_root = self.data_root if self.data_root[-1] == '/' else self.data_root + '/'
        for output in outputs:
            preds.append(output['poses'].cpu().numpy()[:, :self.num_joints])
            vis.append(output['vis'].cpu().numpy()[:, :self.num_joints])
            scores.append(output['scores'])
            image_paths.append(output['image_paths'][0].replace(data_root, ''))

        kpts = defaultdict(list)
        # iterate over images
        for idx, _preds in enumerate(preds):
            str_image_path = image_paths[idx]
            image_id = self.name2id[str_image_path]
            # iterate over people
            for idx_person, kpt in enumerate(_preds):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (
                        np.max(kpt[:, 1]) - np.min(kpt[:, 1]))

                kpts[image_id].append({
                    'keypoints': kpt[:, 0:3],
                    'score': scores[idx][idx_person],
                    'vis': vis[idx][idx_person],
                    'image_id': image_id,
                    'area': area,
                })

        valid_kpts = list(kpts.values())

        results = self._write_coco_keypoint_results(valid_kpts, res_file)

        info_str = self.do_python_keypoint_eval(results, eval_mode=eval_mode)
        name_value = OrderedDict(info_str)
        return name_value

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [{
            'cat_id': 1,
            'cls_ind': 0,
            'cls': 'person',
            'ann_type': 'keypoints',
            'keypoints': keypoints
        }]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        os.makedirs(os.path.dirname(res_file), exist_ok=True)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        return results

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                             self.num_joints * 3)

            for img_kpt, key_point in zip(img_kpts, key_points):
                kpt = key_point.reshape((self.num_joints, 3))
                left_top = np.amin(kpt, axis=0)
                right_bottom = np.amax(kpt, axis=0)

                w = right_bottom[0] - left_top[0]
                h = right_bottom[1] - left_top[1]

                cat_results.append({
                    'image_id': img_kpt['image_id'],
                    'category_id': cat_id,
                    'keypoints': key_point.tolist(),
                    'score': float(img_kpt['score']),
                    'bbox': np.array([left_top[0], left_top[1], w, h]).tolist()
                })

        return cat_results

    def do_python_keypoint_eval(self, results, eval_mode='all'):
        if isinstance(results, str):
            if os.path.isdir(results):
                results = os.path.join(results, 'result_keypoints.json')
            with open(results) as f:
                results = json.load(f)

        name2pred = {}
        id2res = defaultdict(list)
        for res in results:
            id2res[res['image_id']].append(res)

        for img_id in self.img_ids:
            res = id2res[img_id]
            img_info = self.coco.imgs[img_id]
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            ann_info_parsed = self._parse_ann_info(self.coco.load_imgs([img_id])[0], ann_info)
            cam = ann_info_parsed['cam']
            cam = {k: np.array(v) for k, v in cam.items()}
            norm_depth = np.sqrt(cam['K'][0, 0]*cam['K'][1, 1])
            if len(res) == 0:
                pred = np.zeros([1, self.num_joints, 3])
            else:
                pred_img = np.array([x['keypoints'] for x in res]).reshape(len(res), -1, 3)[:, :self.num_joints]
                if self.norm_depth:
                    if self.abs_dz:
                        root_depth = pred_img[:, [self.ROOT_IDX], 2]
                        dz = pred_img[..., 2] - root_depth
                        pred_img[..., 2] = root_depth * norm_depth + dz
                        pred_img[..., 2] *= self.depth_factor
                    else:
                        pred_img[..., 2] *= norm_depth * self.depth_factor
                pred = pixel2world(pred_img.reshape(-1, 3).T, cam['K'], cam['R'], cam['t'])[-1].T.reshape(
                    pred_img.shape)
            # pred = pred_img
            name2pred[img_info['file_name']] = pred

        res_dict = Manager().dict()
        pool = []
        eval_ids = list(range(20))
        for ts in eval_ids:
            p = Process(target=eval_mupots_abs, args=(ts, self.data_root, name2pred, res_dict, eval_mode))
            p.start()
            pool.append(p)
        for p in pool:
            p.join()

        sequencewise_per_joint_error = sum([res_dict[k]['sequencewise_per_joint_error'] for k in eval_ids], [])
        pck_curve_array, pck_array, auc_array = calculate_multiperson_errors(sequencewise_per_joint_error)

        sequencewise_per_joint_error_abs = sum([res_dict[k]['sequencewise_per_joint_error_abs'] for k in eval_ids], [])
        _, pck_array_abs, _ = calculate_multiperson_errors(sequencewise_per_joint_error_abs)

        output_root = []

        pck_mean = sum([i[-1] for i in pck_array]) / len(pck_array)
        pck_mean_abs = sum([i[-1] for i in pck_array_abs]) / len(pck_array_abs)

        output = [('PCK_MEAN:', f'{pck_mean*100:.2f}'), ('PCK_MEAN_ABS:', f'{pck_mean_abs*100:.2f}'), ]

        return output + output_root


def load_annot(fname):
    def parse_pose(dt):
        res = {}
        annot2 = dt['annot2'][0, 0]
        annot3 = dt['annot3'][0, 0]
        annot3_univ = dt['univ_annot3'][0, 0]
        is_valid = dt['isValidFrame'][0, 0][0, 0]
        res['annot2'] = annot2
        res['annot3'] = annot3
        res['annot3_univ'] = annot3_univ
        res['is_valid'] = is_valid
        return res

    data = sio.loadmat(fname)['annotations']
    results = []
    num_frames, num_inst = data.shape[0], data.shape[1]
    for j in range(num_inst):
        buff = []
        for i in range(num_frames):
            buff.append(parse_pose(data[i, j]))
        results.append(buff)
    return results


def load_occ(fname):
    data = sio.loadmat(fname)['occlusion_labels']
    results = []
    num_frames, num_inst = data.shape[0], data.shape[1]
    for i in range(num_frames):
        buff = []
        for j in range(num_inst):
            buff.append(data[i][j])
        results.append(buff)
    return results


def mpii_joint_groups():
    joint_groups = [
        ['Head', [0]],
        ['Neck', [1]],
        ['Shou', [2, 5]],
        ['Elbow', [3, 6]],
        ['Wrist', [4, 7]],
        ['Hip', [8, 11]],
        ['Knee', [9, 12]],
        ['Ankle', [10, 13]],
    ]
    all_joints = []
    for i in joint_groups:
        all_joints += i[1]
    return joint_groups, all_joints


def mpii_get_joints(set_name):
    original_joint_names = ['spine3', 'spine4', 'spine2', 'spine1', 'spine',
                            'neck', 'head', 'head_top', 'left_shoulder', 'left_arm', 'left_forearm',
                            'left_hand', 'left_hand_ee', 'right_shoulder', 'right_arm', 'right_forearm', 'right_hand',
                            'right_hand_ee', 'left_leg_up', 'left_leg', 'left_foot', 'left_toe', 'left_ee',
                            'right_leg_up', 'right_leg', 'right_foot', 'right_toe', 'right_ee']

    all_joint_names = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis',
                       'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow',
                       'left_wrist', 'left_hand', 'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',
                       'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',
                       'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe']

    if set_name == 'relavant':
        joint_idx = [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]
        joint_parents_o1 = [2, 16, 2, 3, 4, 2, 6, 7, 15, 9, 10, 15, 12, 13, 15, 15, 2]
        joint_parents_o2 = [16, 15, 16, 2, 3, 16, 2, 6, 16, 15, 9, 16, 15, 12, 15, 15, 16]
        joint_idx = [i - 1 for i in joint_idx]
        joint_parents_o1 = [i - 1 for i in joint_parents_o1]
        joint_parents_o2 = [i - 1 for i in joint_parents_o2]
        joint_names = [all_joint_names[i] for i in joint_idx]
        return joint_idx, joint_parents_o1, joint_parents_o2, joint_names
    else:
        raise NotImplementedError('Not implemented yet.')


def mean(l):
    return sum(l) / len(l)


def mpii_compute_3d_pck(seq_err):
    pck_curve_array = []
    pck_array = []
    auc_array = []
    thresh = np.arange(0, 200, 5)
    pck_thresh = 150
    joint_groups, all_joints = mpii_joint_groups()
    for seq_idx in range(len(seq_err)):
        pck_curve = []
        pck_seq = []
        auc_seq = []
        err = np.array(seq_err[seq_idx]).astype(np.float32)

        for j in range(len(joint_groups)):
            err_selected = err[:, joint_groups[j][1]]
            buff = []
            for t in thresh:
                pck = np.float32(err_selected < t).sum() / len(joint_groups[j][1]) / len(err)
                buff.append(pck)  # [Num_thresholds]
            pck_curve.append(buff)
            auc_seq.append(mean(buff))
            pck = np.float32(err_selected < pck_thresh).sum() / len(joint_groups[j][1]) / len(err)
            pck_seq.append(pck)

        buff = []
        for t in thresh:
            pck = np.float32(err[:, all_joints] < t).sum() / len(err) / len(all_joints)
            buff.append(pck)  # [Num_thresholds]
        pck_curve.append(buff)

        pck = np.float32(err[:, all_joints] < pck_thresh).sum() / len(err) / len(all_joints)
        pck_seq.append(pck)

        pck_curve_array.append(pck_curve)  # [num_seq: [Num_grpups+1: [Num_thresholds]]]
        pck_array.append(pck_seq)  # [num_seq: [Num_grpups+1]]
        auc_array.append(auc_seq)  # [num_seq: [Num_grpups]]

    return pck_curve_array, pck_array, auc_array


def calculate_multiperson_errors(seq_err):
    return mpii_compute_3d_pck(seq_err)


def norm_by_bone_length(pred, gt, o1, trav):
    mapped_pose = pred.copy()

    for i in range(len(trav)):
        idx = trav[i]
        gt_len = np.linalg.norm(gt[:, idx] - gt[:, o1[i]])
        pred_vec = pred[:, idx] - pred[:, o1[i]]
        pred_len = np.linalg.norm(pred_vec)
        mapped_pose[:, idx] = mapped_pose[:, o1[i]] + pred_vec * gt_len / pred_len
    return mapped_pose


def procrustes(predicted, target):
    predicted = predicted.T
    target = target.T
    predicted = predicted[None, ...]
    target = target[None, ...]

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t
    return predicted_aligned[0].T


def match(pose1, pose2, o1=None, trav=None, threshold=250):
    joint_groups, all_joints = mpii_joint_groups()
    matches = []
    matches_abs = []
    p2 = np.float32(pose2)
    if o1 is not None:
        p2_root = p2[:, :, 14:15]
        p2 = p2 - p2_root
    for i in range(len(pose1)):
        p1 = np.float32(pose1[i])
        p1_root = p1[:, 14:15]
        p1 = p1 - p1_root
        diffs = []
        diffs_abs = []
        for j in range(len(p2)):
            p = p2[j].copy()
            depth_ratio = p1_root[[2]] / p2_root[j, [2]]
            p[:2] *= depth_ratio
            p = norm_by_bone_length(p, p1, o1, trav)
            diff = np.sqrt(np.power(p - p1, 2).sum(axis=0)).mean()
            diff_abs = np.sqrt(np.power(p + p2_root[j] - p1 - p1_root, 2).sum(axis=0)).mean()
            diffs.append(diff)
            diffs_abs.append(diff_abs)
        diffs = np.float32(diffs)
        diffs_abs = np.float32(diffs_abs)
        idx = np.argmin(diffs)
        if diffs.min() > threshold:
            matches.append(-1)
        else:
            matches.append(idx)
        idx = np.argmin(diffs_abs)
        if diffs_abs.min() > threshold:
            matches_abs.append(-1)
        else:
            matches_abs.append(idx)
    return matches, matches_abs


def eval_mupots_abs(ts, test_annot_base, name2pred, res_dict, eval_mode='all'):
    _, o1, o2, relevant_labels = mpii_get_joints('relavant')
    num_joints = len(o1)

    evaluation_mode = 0 if eval_mode == 'all' else 1
    # 0 for all, 1 for matched
    safe_traversal_order = [15, 16, 2, 1, 17, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    safe_traversal_order = [i - 1 for i in safe_traversal_order]

    # eval sequence ts
    # print('Seq:', ts + 1)
    sequencewise_per_joint_error = []
    sequencewise_per_joint_error_abs = []
    sequencewise_frames = []

    annots = load_annot(os.path.join(test_annot_base, 'TS%d/annot.mat' % (ts + 1)))
    occlusions = load_occ(os.path.join(test_annot_base, 'TS%d/occlusion.mat' % (ts + 1)))
    num_frames = len(annots[0])
    num_person = len(annots)
    undetected_people = 0
    annotated_people = 0
    pje = []
    pje_correct = []
    pje_abs = []
    pje_correct_abs = []
    pjocc = []
    pjvis = []
    sequencewise_frames.append(num_frames)
    for i in range(num_frames):
        valid_annotations = 0;
        for k in range(num_person):
            # print(annots[k][i])
            if annots[k][i]['is_valid'] == 1:
                valid_annotations += 1
        annotated_people += valid_annotations
        if valid_annotations == 0:
            continue

        gt_p2d = []
        gt_p3d = []
        gt_vis = []
        gt_occ = []
        gt_pose_vis = []
        matching_joints = list(range(1, 14))
        for k in range(num_person):
            if annots[k][i]['is_valid'] == 1:
                gt_p2d.append(annots[k][i]['annot2'][:, matching_joints])
                gt_p3d.append(annots[k][i]['annot3'])
                gt_vis.append(np.ones([1, len(matching_joints)]))
                gt_occ.append(occlusions[i][k])
                gt_pose_vis.append(1 - gt_occ[-1])

        filename = 'TS%d/img_%06d.jpg' % (ts + 1, i)
        pred_p3d = name2pred[filename]
        pred_p3d = pred_p3d.transpose(0, 2, 1)

        invalid = pred_p3d[:, 2, 14] == 0
        if invalid.sum() > 0:
            pred_p3d = pred_p3d[~invalid]
        if len(pred_p3d) == 0:
            pred_p3d = np.zeros((1, 3, 17))

        matches, matches_abs = match(gt_p3d, pred_p3d, o1, safe_traversal_order[1:])
        for k in range(len(matches)):
            pred_considered = 0
            gtP_abs = gt_p3d[k]
            gtP = gt_p3d[k] - gt_p3d[k][:, 14:15]
            if matches[k] != -1:
                predP_abs = pred_p3d[matches[k]]
                pred_root = predP_abs[:, 14:15]
                predP = predP_abs - pred_root
                depth_ratio = gtP_abs[[2], [14]] / predP_abs[[2], [14]]
                predP[:2] = predP[:2] * depth_ratio
                predP_align = procrustes(predP, gtP)
                predP = norm_by_bone_length(predP, gtP, o1, safe_traversal_order[1:])

                # if matches_abs[k] != -1:
                #     predP_abs = pred_p3d[matches_abs[k]]
                p = predP_abs - predP_abs[:, 14:15]
                depth_ratio = gtP_abs[[2], [14]] / predP_abs[[2], [14]]
                p[:2] = p[:2] * depth_ratio
                p_align = procrustes(p, gtP)
                p = norm_by_bone_length(p, gtP, o1, safe_traversal_order[1:])
                predP_abs = p + predP_abs[:, 14:15]
                predP_abs_align = p_align - p_align[:, 14:15] + predP_abs[:, 14:15]

                pred_considered = 1
            else:
                undetected_people += 1
                predP = predP_abs = predP_align = predP_abs_align = 100000 * np.ones(gtP.shape)
                if evaluation_mode == 0:
                    pred_considered = 1

            if pred_considered == 1:
                errorP = np.sqrt(np.power(predP - gtP, 2).sum(axis=0))
                errorP_correct = np.sqrt(np.power(predP_align - gtP, 2).sum(axis=0))
                errorP_abs = np.sqrt(np.power(predP_abs - gtP_abs, 2).sum(axis=0))
                errorP_correct_abs = np.sqrt(np.power(predP_abs_align - gtP_abs, 2).sum(axis=0))
                pje.append(errorP)  # num_tested poses
                pje_correct.append(errorP_correct)
                pje_abs.append(errorP_abs)  # num_tested poses
                pje_correct_abs.append(errorP_correct_abs)
                pjocc.append(gt_occ[k])
                pjvis.append(gt_vis[k])

    sequencewise_per_joint_error.append(pje)
    sequencewise_per_joint_error_abs.append(pje_abs)

    res_dict[ts] = dict(
        sequencewise_per_joint_error=sequencewise_per_joint_error,
        sequencewise_per_joint_error_abs=sequencewise_per_joint_error_abs,
    )

    return sequencewise_per_joint_error

