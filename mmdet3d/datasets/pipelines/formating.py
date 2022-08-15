import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor

PIPELINES._module_dict.pop('DefaultFormatBundle')


@PIPELINES.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = DC(to_tensor(img), stack=True)
        for key in [
            'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
            'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
            'pts_semantic_mask', 'centers2d', 'depths'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        if 'gt_bboxes_3d' in results:
            if isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = DC(
                    results['gt_bboxes_3d'], cpu_only=True)
            else:
                results['gt_bboxes_3d'] = DC(
                    to_tensor(results['gt_bboxes_3d']))

        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Collect3D(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'transform_mat', 'pcd_rot', 'cam'),
                 num_joints=None,
                 debug=False):
        self.keys = keys
        self.meta_keys = meta_keys
        self.num_joints = num_joints
        self.debug = debug

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
            # else:
            #     print('WARNING: key "{}" not in input_dict!'.format(key))

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]

        if self.debug:
            # import IPython; IPython.embed()
            flip = 'flip_' if results['flip'] else 'noflip_'
            img = results['img'].data.permute(1, 2, 0).numpy()
            norm_cfg = results['img_norm_cfg']
            mean, std, to_rgb = norm_cfg['mean'], norm_cfg['std'], norm_cfg['to_rgb']
            img = img * std + mean
            if to_rgb:
                img = img[..., ::-1]

            import matplotlib.pyplot as plt
            import cv2
            import os
            # draw 2d keypoints
            os.makedirs('debug', exist_ok=True)

            def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1, color=None):
                # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
                cmap = plt.get_cmap('rainbow')
                colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
                colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
                # Perform the drawing on a copy of the image, to allow for blending.
                kp_mask = np.copy(img)
                # Draw the keypoints.
                for link in range(len(kps_lines)):
                    i1 = kps_lines[link][0]
                    i2 = kps_lines[link][1]
                    p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
                    p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
                    if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                        cv2.line(
                            kp_mask, p1, p2,
                            color=colors[link] if color is None else color, thickness=2, lineType=cv2.LINE_AA)
                    # right: red, left: blue
                    lr_color = lambda x: (0, 0, 255) if x >= 9 else (255, 0, 0)
                    if color is not None:
                        lr_color = lambda x: color
                    if kps[2, i1] > kp_thresh:
                        cv2.circle(
                            kp_mask, p1,
                            radius=2, color=lr_color(i1), thickness=-1, lineType=cv2.LINE_AA)
                    if kps[2, i2] > kp_thresh:
                        cv2.circle(
                            kp_mask, p2,
                            radius=2, color=lr_color(i2), thickness=-1, lineType=cv2.LINE_AA)
                # Blend the keypoints.
                return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

            LIMBS15 = [[0, 1],
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

            LIMBS17 = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11],
                       [11, 13], [13, 15],
                       [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]
            LIMBS21 = (
                (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13),
                (13, 20),
                (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))
            LIMBS = eval('LIMBS%d' % self.num_joints)
            tmpimg = img.copy().clip(0, 255).astype(np.uint8)
            tmpkps = np.zeros((3, self.num_joints))
            gt_poses_3d = results['gt_poses_3d'].data.numpy()
            gt_bboxes = results['gt_bboxes'].data.numpy()
            centers2d = gt_poses_3d[:, :2]
            joints_img = gt_poses_3d[:, 3:3 + self.num_joints * 3].reshape(-1, self.num_joints, 3)
            joints_vis = gt_poses_3d[:, 3 + self.num_joints * 3:]

            num_person = len(centers2d)
            # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
            cmap = plt.get_cmap('rainbow')
            colors = [cmap(i) for i in np.linspace(0, 1, num_person)]
            colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
            colors = [list(map(int, x)) for x in colors]
            # colors = [''.join(['#', *tuple('%02X' % t for t in x)]) for x in colors]

            for i in range(len(centers2d)):
                joints2d = joints_img[i]
                visible = joints_vis[i]
                center = centers2d[i]
                # print(joints2d)
                tmpkps[0, :], tmpkps[1, :] = joints2d[:, 0], joints2d[:, 1]
                tmpkps[2, :] = visible
                tmpimg = vis_keypoints(tmpimg, tmpkps, LIMBS, color=colors[i])

                cv2.circle(
                    tmpimg, (int(center[0]), int(center[1])),
                    radius=5, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                x1, y1, x2, y2 = gt_bboxes[i]
                cv2.rectangle(tmpimg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            out_name = os.path.join('debug', flip + '{}'.format(results['img_info']['file_name'].split('/')[-1]))
            cv2.imwrite(out_name, tmpimg)
            print('DRAW')
            # input()
            import ipdb;
            ipdb.set_trace()
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'


@PIPELINES.register_module()
class DefaultFormatBundle3D(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        super(DefaultFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = DC(results['points'].tensor)

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

        if self.with_gt:
            # Clean GT bboxes in the final
            if 'gt_bboxes_3d_mask' in results:
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][
                    gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][
                        gt_bboxes_3d_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][
                        gt_bboxes_3d_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                 dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                        dtype=np.int64)
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n)
                        for n in results['gt_names_3d']
                    ],
                        dtype=np.int64)
        results = super(DefaultFormatBundle3D, self).__call__(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str


@PIPELINES.register_module()
class DefaultFormatBundlePose3D(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        super(DefaultFormatBundlePose3D, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = DC(to_tensor(img), stack=True)
        for key in [
            'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_poses_3d',
            'gt_labels_3d', 'centers2d', 'depths', 'transform_mat'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str
