import os
import copy
import mmcv
import json
import numpy as np
from collections import defaultdict, OrderedDict
from mmdet.datasets import DATASETS, CocoDataset
from mytools.vis_3d import pixel2world


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


@DATASETS.register_module()
class CMUPanopticDataset(CocoDataset):
    CLASSES = ('person',)
    # moputs to panoptic [1, 0, 14, 5, 6, 7, 11, 12, 13, 2, 3, 4, 8, 9, 10]
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
    skeleton = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10],
                [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14]]
    ROOT_IDX = 2

    def __init__(self,
                 data_root,
                 load_interval=1,
                 use_bbox_center=False,
                 norm_depth=True,
                 abs_dz=True,
                 depth_factor=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_joints = len(self.JOINTS_DEF)
        self.data_root = data_root
        self.load_interval = load_interval
        self.norm_depth = norm_depth
        self.depth_factor = depth_factor
        self.abs_dz = abs_dz
        if abs_dz:
            assert norm_depth

        self.name2id = {}
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            self.name2id[os.path.basename(info['file_name'])] = i
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
        try:
            if len(ann_info['gt_poses_3d']) == 0:
                return None
            results = dict(img_info=img_info, ann_info=ann_info)
            if self.proposals is not None:
                results['proposals'] = self.proposals[idx]
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

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """

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
        """Parse bbox annotation.

        Args:
            img_info (list[dict]): Image info.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, \
            labels, gt_poses_3d, gt_labels_3d, centers2d, depths, bboxes_ignore
        """
        K = img_info['cam']['K']
        f = np.sqrt(K[0][0] * K[1][1])
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
                # 3D pose
                pose_3d = ann['joints3d_img']
                pose_vis = ann['joints2d_vis']
                pose_3d = np.array(pose_3d)
                if self.norm_depth:
                    pose_3d[:, 2] /= self.depth_factor
                    if self.abs_dz:
                        abs_dz = pose_3d[:, 2] - pose_3d[[self.ROOT_IDX], 2]
                    pose_3d[:, 2] /= f
                if pose_3d.max() - pose_3d.min() < 10:
                    continue
                if not self.use_bbox_center:
                    if pose_vis[self.ROOT_IDX][0] == 0:
                        gt_bboxes_ignore.append(bbox)
                        continue
                    # c2d = ann['center2d']
                    c2d = pose_3d[self.ROOT_IDX].copy()
                else:
                    c2d = pose_3d[self.ROOT_IDX].copy()
                    # c2d = np.array(ann['center2d']).copy()
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
                    np.array(pose_vis, dtype=np.float)[:, 0].reshape(-1)
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
            gt_poses_3d = np.zeros((0, 3+self.num_joints*3+self.num_joints), dtype=np.float32)
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
        if 'cam' in img_info:
            ann['cam'] = img_info['cam']

        return ann


    def evaluate(self, outputs, res_folder='tmp', metric='mpjpe', **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mpjpe']
        for metric in metrics:
            if metric.lower() not in allowed_metrics:
                raise KeyError(f'metric {metric.lower()} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')

        preds = []
        vis = []
        scores = []
        image_paths = []

        for output in outputs:
            preds.append(output['poses'].cpu().numpy())
            vis.append(output['vis'].cpu().numpy())
            scores.append(output['scores'])
            image_paths.append(output['image_paths'][0])

        kpts = defaultdict(list)
        # iterate over images
        for idx, _preds in enumerate(preds):
            str_image_path = image_paths[idx]
            image_id = self.name2id[os.path.basename(str_image_path)]
            # iterate over people
            for idx_person, kpt in enumerate(_preds):
                # use bbox area
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

        info_str = self.do_python_keypoint_eval(results)
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

    def vectorize_distance(self, preds, gts, vis):
        mse = np.sqrt(((gts[:, None] - preds[None]) ** 2).sum(axis=-1))
        mse = mse * vis[:, None]
        dist = mse.mean(-1)
        min_ids = dist.argmin(1)
        return min_ids

    def mse(self, preds, gts, vis):
        assert preds.shape == gts.shape == (*vis.shape, 3), (preds.shape, gts.shape, vis.shape)
        return np.sqrt(((preds[vis > 0] - gts[vis > 0]) ** 2).sum(axis=-1))

    def do_python_keypoint_eval(self, results):
        if isinstance(results, str):
            if os.path.isdir(results):
                results = os.path.join(results, 'result_keypoints.json')
            with open(results) as f:
                results = json.load(f)
        avg_meter = AverageMeter('P1', ':.2f')
        all_pose = np.array([x['joints3d'] for x in self.coco.anns.values()], dtype=np.float) / 10
        all_vis = np.array([x['joints3d_vis'] for x in self.coco.anns.values()], dtype=np.float)
        all_pose = all_pose - all_pose[:, [self.ROOT_IDX], :]
        mean_pose = (all_pose * all_vis).sum(0) / all_vis.sum(0)
        mean_pose[np.isnan(mean_pose)] = 0
        for img_id in self.img_ids:
            res = [x for x in results if x['image_id'] == img_id]
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            ann_info_parsed = self._parse_ann_info(self.coco.load_imgs([img_id])[0], ann_info)
            cam = ann_info_parsed['cam']
            cam = {k:np.array(v) for k, v in cam.items()}
            norm_depth = np.sqrt(cam['K'][0, 0]*cam['K'][1, 1])
            pred_img = np.array([x['keypoints'] for x in res]).reshape(-1, self.num_joints, 3)
            if self.norm_depth:
                # rescale predicted human depth according to image size
                if self.abs_dz:
                    root_depth = pred_img[:, [self.ROOT_IDX], 2]
                    dz = pred_img[..., 2] - root_depth
                    pred_img[..., 2] = root_depth * norm_depth + dz
                    pred_img[..., 2] *= self.depth_factor
                else:
                    pred_img[..., 2] *= norm_depth * self.depth_factor
            pred = pixel2world(pred_img.reshape(-1, 3).T, cam['K'], cam['R'], cam['t'])[-1].T.reshape(pred_img.shape)
            gt_img = ann_info_parsed['gt_poses_3d'][:, 3:3+self.num_joints*3].reshape(-1, self.num_joints, 3)
            if self.norm_depth and self.abs_dz:
                depth = ann_info_parsed['gt_poses_3d'][:, [2]]
                depth = depth * norm_depth
                gt_img[..., 2] += depth
            gt = pixel2world(gt_img.reshape(-1, 3).T, cam['K'], cam['R'], cam['t'])[-1].T.reshape(gt_img.shape)
            gt_vis = ann_info_parsed['gt_poses_3d'][:, 3+self.num_joints*3:]
            if len(gt) == 0:
                continue
            # root-aligned MPJPE
            pred = pred - pred[:, [self.ROOT_IDX]]
            if len(pred) == 0:
                # if there is no valid prediction, use mean human pose instead
                pred = np.concatenate([pred, mean_pose[None]])
            gt = gt - gt[:, [self.ROOT_IDX]]
            paired_idxs = self.vectorize_distance(pred, gt, gt_vis)
            jpe = self.mse(pred[paired_idxs], gt, gt_vis)
            mpjpe = jpe.mean() * 10 # cm to mm
            if len(jpe) > 0:
                avg_meter.update(mpjpe, n=len(gt))

        return [['MPJPE:', f'{avg_meter.avg:.2f}mm']]
