import os
import copy
import mmcv
import json
import numpy as np
import base64
import cv2
from collections import defaultdict, OrderedDict

from mmdet.datasets import DATASETS, CocoDataset
from mmdet3d.utils.tsv_file import TSVFile, CompositeTSVFile
from mmdet3d.utils.tsv_file_ops import load_linelist_file, load_from_yaml_file, find_file_path_in_yaml


def img_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring)
        nparr = np.frombuffer(jpgbytestring, np.uint8)
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return r
    except ValueError:
        return None


@DATASETS.register_module()
class MuCo3DHPDataset(CocoDataset):
    CLASSES = ('person',)
    muco_joint_num = 21
    muco_joints_name = (
        'Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee',
        'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
    muco_flip_pairs = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20))
    muco_skeleton = (
        (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20),
        (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))
    JOINTS_DEF = {k: i for i, k in enumerate(muco_joints_name)}
    ROOT_IDX = muco_joints_name.index('Pelvis')

    def __init__(self,
                 yaml_file,
                 ann_file,
                 pipeline,
                 use_bbox_center=False,
                 use_meshtransformer_ann=False,
                 norm_depth=False,
                 depth_factor=1,
                 abs_dz=False,
                 **kwargs):
        self.use_meshtransformer_ann = use_meshtransformer_ann
        if self.use_meshtransformer_ann:
            pipeline = [x for x in pipeline if x['type'] != 'LoadImageFromFile']
        # MeshTransformer
        if use_meshtransformer_ann:
            self.cfg = load_from_yaml_file(yaml_file)
            self.is_composite = self.cfg.get('composite', False)
            self.root = os.path.dirname(yaml_file)
            if not self.is_composite:
                img_file = find_file_path_in_yaml(self.cfg['img'], self.root)
                label_file = find_file_path_in_yaml(self.cfg.get('label', None), self.root)
                hw_file = find_file_path_in_yaml(self.cfg.get('hw', None), self.root)
                linelist_file = find_file_path_in_yaml(self.cfg.get('linelist', None), self.root)
            else:
                img_file = self.cfg['img']
                hw_file = self.cfg['hw']
                label_file = self.cfg.get('label', None)
                linelist_file = find_file_path_in_yaml(self.cfg.get('linelist', None), self.root)

            self.img_tsv = self.get_tsv_file(img_file)
            self.label_tsv = None if label_file is None else self.get_tsv_file(label_file)
            self.hw_tsv = None if hw_file is None else self.get_tsv_file(hw_file)
            if self.is_composite:
                assert os.path.isfile(linelist_file)
                self.line_list = [i for i in range(self.hw_tsv.num_rows())]
            else:
                self.line_list = load_linelist_file(linelist_file)
            self.imgkey2idx = self.prepare_image_key_to_index()
        self.norm_depth = norm_depth
        self.depth_factor = depth_factor
        self.abs_dz = abs_dz
        if abs_dz:
            assert norm_depth

        # COCO
        super().__init__(ann_file, pipeline, **kwargs)
        self.num_joints = len(self.JOINTS_DEF)

        self.name2id = {}
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            self.name2id[os.path.basename(info['file_name'])] = i
        self.use_bbox_center = use_bbox_center

        if self.test_mode:
            valid_inds = self._filter_imgs()
            valid_inds = valid_inds[::4]
            self.data_infos = [self.data_infos[i] for i in valid_inds]

    def _filter_imgs(self, min_size=32):
        if self.use_meshtransformer_ann:
            """Filter images too small or without ground truths."""
            valid_inds = []
            # obtain images that contain annotation
            ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
            # obtain images that contain annotations of the required categories
            ids_in_cat = set()
            for i, class_id in enumerate(self.cat_ids):
                ids_in_cat |= set(self.coco.cat_img_map[class_id])
            # merge the image id sets of the two conditions and use the merged set
            # to filter out images if self.filter_empty_gt=True
            ids_in_cat &= ids_with_ann

            invalid_names = []
            invalid_count = 0
            valid_img_ids = []
            for i, img_info in enumerate(self.data_infos):
                img_id = self.img_ids[i]
                if self.filter_empty_gt and img_id not in ids_in_cat:
                    continue
                if img_info['file_name'] not in self.imgkey2idx:
                    invalid_count += 1
                    invalid_names.append(img_info['file_name'])
                    continue
                if min(img_info['width'], img_info['height']) >= min_size:
                    valid_inds.append(i)
                    valid_img_ids.append(img_id)
            self.img_ids = valid_img_ids

            print('FILTERED IMAGE NUM:', invalid_count)
            print('TOTAL IMAGE NUM:', len(valid_inds))
        else:
            return super(MuCo3DHPDataset, self)._filter_imgs(min_size)
        return valid_inds

    def get_tsv_file(self, tsv_file):
        if tsv_file:
            if self.is_composite:
                return CompositeTSVFile(tsv_file, self.linelist_file,
                                        root=self.root)
            tsv_path = find_file_path_in_yaml(tsv_file, self.root)
            return TSVFile(tsv_path)

    def get_valid_tsv(self):
        # sorted by file size
        if self.hw_tsv:
            return self.hw_tsv
        if self.label_tsv:
            return self.label_tsv

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.get_key(i): i for i in range(tsv.num_rows())}

    def get_line_no(self, idx):
        return idx if self.line_list is None else self.line_list[idx]

    def get_image(self, idx):
        line_no = self.get_line_no(idx)
        row = self.img_tsv[line_no]
        # use -1 to support old format with multiple columns.
        cv2_im = img_from_base64(row[-1])
        cv2_im = cv2_im.astype(np.float32, copy=True)
        return cv2_im

    def get_label(self, idx):
        line_no = self.get_line_no(idx)
        row = self.label_tsv[line_no][-1]
        return json.loads(row)

    def _load_image_from_file(self, results):
        filename = results['img_info']['filename']
        img = self.get_image(self.imgkey2idx[filename])
        shape = (results['img_info']['height'], results['img_info']['width'])
        if img.shape[:2] != shape:
            # import ipdb; ipdb.set_trace()
            img = cv2.resize(img, shape[::-1], interpolation=cv2.INTER_LINEAR)
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

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
        try:
            if len(ann_info['gt_poses_3d']) == 0:
                return None
            results = dict(img_info=img_info, ann_info=ann_info)
            if self.proposals is not None:
                results['proposals'] = self.proposals[idx]
            self.pre_pipeline(results)
            if self.use_meshtransformer_ann:
                results = self._load_image_from_file(results)
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
        if self.use_meshtransformer_ann:
            results = self._load_image_from_file(results)
        return self.pipeline(results)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox annotation.

        Args:
            img_info (list[dict]): Image info.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, \
            labels, gt_poses_3d, gt_labels_3d, centers2d, depths, bboxes_ignore
        """
        # generate pseudo camera parameters
        f = img_info['f']
        c = img_info['c']
        cam = dict(
            K=[[f[0], 0., c[0]],
               [0., f[1], c[1]]],
            R=[[1.0, 0.0, 0.0],
               [0.0, 0.0, -1.0],
               [0.0, 1.0, 0.0]],
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
                # from mytools.vis_3d import pixel2world, world2pixel
                # pose_img_ = world2pixel(pose_cam.T, cam['K'], cam['R'], cam['t'])[-1].T
                # import ipdb; ipdb.set_trace()
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
                    if pose_vis[self.ROOT_IDX] == 0:
                        gt_bboxes_ignore.append(bbox)
                        continue
                    c2d = pose_3d[self.ROOT_IDX].copy()
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
            if not self.test_mode:
                return None

        if gt_poses_3d[:, 3+self.num_joints*3:].sum() < 6 and not self.test_mode:
            return None

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

    def evaluate(self):
        raise NotImplementedError
