import copy
import mmcv
import numpy as np
from mmdet.datasets import DATASETS, CocoDataset
import os
from collections import defaultdict, OrderedDict
import json
from xtcocotools.cocoeval import COCOeval
from xtcocotools.coco import COCO
from mmdet3d.core.post_processing import oks_nms, soft_oks_nms
import torch.optim

@DATASETS.register_module()
class COCOKeypointsDataset(CocoDataset):

    # muco_joint_num = 21
    # muco_joints_name = (
    #     'Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee',
    #     'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
    # convert_ids = [-1, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1]
    # compose_ids = {
    #     1:  [5, 6],     # thorax -> coco:[l_shoulder, r_shoulder]
    #     14: [11, 12],   # pelvis -> coco:[l_hip, r_hip]
    #     15: [1*, 14*],  # Spine  -> [thorax, pelvis]
    # }

    # panoptic:
    #     'neck': 0,
    #     'nose': 1,
    #     'mid-hip': 2,
    #     'l-shoulder': 3,
    #     'l-elbow': 4,
    #     'l-wrist': 5,
    #     'l-hip': 6,
    #     'l-knee': 7,
    #     'l-ankle': 8,
    #     'r-shoulder': 9,
    #     'r-elbow': 10,
    #     'r-wrist': 11,
    #     'r-hip': 12,
    #     'r-knee': 13,
    #     'r-ankle': 14,
    # convert_ids = [-1, 0, -1, 5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16]

    CLASSES = ('person',)
    JOINTS_DEF = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    # 'https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/'
    # 'pycocotools/cocoeval.py#L523'
    sigmas = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
        .87, .87, .89, .89
    ]) / 10.0

    def __init__(self,
                 data_root,
                 load_interval=1,
                 use_nms=False,
                 use_bbox_center=False,
                 convert_ids=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_joints = len(self.JOINTS_DEF)
        self.data_root = data_root
        self.load_interval = load_interval
        self.convert_ids = convert_ids

        self.name2id = {}
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            self.name2id[os.path.basename(info['file_name'])] = i
        cats = [
            cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self.use_nms = use_nms
        self.use_bbox_center = use_bbox_center

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        if ann_info is None:
            return None
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        try:
            self.pre_pipeline(results)
            return self.pipeline(results)
        except Exception as e:
            s = repr(e)
            if s == 'TypeError("\'NoneType\' object is not subscriptable")':
                return None
            else:
                raise e

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

        results['img_fields'] = []
        results['pose3d_fields'] = []

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox annotation.

        Args:
            img_info (list[dict]): Image info.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, \
            labels, gt_poses_3d, gt_labels_3d, centers2d, depths, bboxes_ignore
        """
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
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                # 2D pose and visible
                keyponts = np.array(ann['keypoints']).reshape(self.num_joints, 3)
                pose_vis = (keyponts[..., 2] > 0).astype(np.float)
                # bbox filter
                bbox_np = np.array(bbox, dtype=np.float).reshape(2, 2)
                bbox_np[:, 0] = bbox_np[:, 0].clip(0, img_info['width'] - 1)
                bbox_np[:, 1] = bbox_np[:, 1].clip(0, img_info['height'] - 1)
                bbox_wh = bbox_np[1, :] - bbox_np[0, :]
                if (bbox_wh < 2).any() or bbox_wh.prod() < 64:
                    continue
                # 3D pose
                pose_3d = keyponts.copy()
                pose_3d[..., 2] = 0

                if not self.use_bbox_center:
                    # if in bbox
                    # "left_hip": 11,
                    # "right_hip": 12,
                    root_joints = keyponts[[11, 12], :2]    # [2,2]
                    if not (((root_joints < bbox_np[1]) & (root_joints > bbox_np[0])).all() and
                           np.abs(root_joints[0, 1] - root_joints[1, 1]) < h / 4):
                        # print(pose_vis[[11,12]])
                        # print(root_joints, bbox_np)
                        if pose_vis[11] == 0 or pose_vis[12] == 0:
                            continue
                    if pose_vis[11] == 0 or pose_vis[12] == 0:
                        continue
                    c2d = 0.5 * (pose_3d[11] + pose_3d[12])
                else:
                    c2d = np.zeros(3, dtype=np.float)
                    c2d[:2] = bbox_np.mean(0)
                center2d = c2d[:2]
                depth = c2d[2]
                general_pose_3d = np.concatenate([
                    np.array(c2d, dtype=np.float).reshape(-1),
                    np.array(pose_3d, dtype=np.float).reshape(-1),
                    np.array(pose_vis, dtype=np.float)
                ])
                # add valid annotation
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
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
            gt_poses_3d = np.zeros((0, 3+self.num_joints*3+self.num_joints), dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)
            return None

        # convert coco-style body joint representation to target style
        if self.convert_ids == 'muco':
            c2d = gt_poses_3d[:, :3]
            uvd = gt_poses_3d[:, 3:3+self.num_joints*3].reshape(-1, self.num_joints, 3)
            vis = gt_poses_3d[:, 3+self.num_joints*3:]
            cids = np.array([-1, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1], dtype=np.long)
            n_pose = uvd.shape[0]
            expanded_uvd = np.zeros((n_pose, len(cids), 3), dtype=np.float32)
            expanded_vis = np.zeros((n_pose, len(cids)), dtype=np.float32)
            expanded_uvd[:, cids >= 0] = uvd[:, cids[cids >= 0]]
            expanded_vis[:, cids >= 0] = vis[:, cids[cids >= 0]]

            # compose_ids = {
            #     1:  [5, 6],     # thorax -> coco:[l_shoulder, r_shoulder]
            #     14: [11, 12],   # pelvis -> coco:[l_hip, r_hip]
            #     15: [1*, 14*],  # Spine  -> [thorax, pelvis]
            # }
            # expanded_uvd[:, 1] = 0.5 * (uvd[:, 5] + uvd[:, 6])
            # expanded_uvd[:, 1] = 0.6 * expanded_uvd[:, 1] + 0.4 * uvd[:, 0]
            # expanded_vis[:, 1] = (vis[:, [5, 6, 0]] == 1).all(axis=1)
            # expanded_uvd[:, 14] = 0.5 * (uvd[:, 11] + uvd[:, 12])
            # expanded_vis[:, 14] = (vis[:, [11, 12]] == 1).all(axis=1)
            # expanded_uvd[:, 15] = 0.5 * (expanded_uvd[:, 1] + expanded_uvd[:, 14])
            # expanded_vis[:, 15] = (expanded_vis[:, [1, 14]] == 1).all(axis=1)

            gt_poses_3d = np.concatenate([c2d, expanded_uvd.reshape(n_pose, -1), expanded_vis], axis=1).astype(np.float32)

            if expanded_vis.sum() < 6: return None
        elif self.convert_ids == 'panoptic':
            c2d = gt_poses_3d[:, :3]
            uvd = gt_poses_3d[:, 3:3+self.num_joints*3].reshape(-1, self.num_joints, 3)
            vis = gt_poses_3d[:, 3+self.num_joints*3:]
            cids = np.array([-1, 0, -1, 5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16], dtype=np.long)
            n_pose = uvd.shape[0]
            expanded_uvd = np.zeros((n_pose, len(cids), 3), dtype=np.float32)
            expanded_vis = np.zeros((n_pose, len(cids)), dtype=np.float32)
            expanded_uvd[:, cids >= 0] = uvd[:, cids[cids >= 0]]
            expanded_vis[:, cids >= 0] = vis[:, cids[cids >= 0]]

            gt_poses_3d = np.concatenate([c2d, expanded_uvd.reshape(n_pose, -1), expanded_vis], axis=1).astype(np.float32)

            if expanded_vis.sum() < 6: return None
        else:
            assert self.convert_ids is None

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

        return ann