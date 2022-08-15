import torch
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
from torch import nn as nn
import numpy as np
import torch.nn.functional as F

from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet3d.core import oks_nms, soft_oks_nms
from .anchor_free_mono3d_pose_head import AnchorFreeMono3DPoseHead
from .real_nvp import RealNVP, RealNVP2D
from .recursive_update import RecursiveUpdateBranch

INF = 1e8


class Bias(nn.Module):
    def __init__(self, bias=0.0, use_bias=False):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float)) if use_bias else 0
        self.use_bias = use_bias

    def forward(self, x):
        if not self.use_bias: return x
        return x + self.bias


@HEADS.register_module()
class DASHead(AnchorFreeMono3DPoseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 center_sample_radius=1.5,
                 centerness_on_reg=True,
                 centerness_branch=(64,),
                 centerness_alpha=2.5,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_reg=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_pose=dict(
                     type='RLELoss3D', residual=True, loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 384),
                                 (384, INF)),
                 recursive_update=None,
                 depth_factor=1,
                 z_norm=1,
                 num_joints=None,
                 root_idx=None,
                 init_cfg=None,
                 **kwargs):
        self.center_sample_radius = center_sample_radius
        self.centerness_on_reg = centerness_on_reg
        self.centerness_branch = centerness_branch
        self.centerness_alpha = centerness_alpha

        self.regress_ranges = regress_ranges
        self.depth_factor = depth_factor
        self.z_norm = z_norm
        self.root_idx = root_idx

        super().__init__(
            num_classes,
            in_channels,
            num_joints=num_joints,
            loss_cls=loss_cls,
            loss_reg=loss_reg,
            loss_pose=loss_pose,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)

        self.loss_centerness = build_loss(loss_centerness)
        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal', name='conv_cls', std=0.01, bias_prob=0.01))

        if 'RLE' in loss_pose['type']:
            self.flow3d = RealNVP()
            self.flow2d = RealNVP2D()
            self.flow3d_update = RealNVP()
            self.flow2d_update = RealNVP2D()

        self.recursive_update = recursive_update
        self.recursive_update_branch = RecursiveUpdateBranch(**recursive_update)

    def _init_pose_convs(self):
        self.pose_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.pose_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls_prev = self._init_branch(
            conv_channels=self.cls_branch,
            conv_strides=(1,) * len(self.cls_branch))
        self.conv_cls = nn.Conv2d(self.cls_branch[-1], self.cls_out_channels, 1)
        # For offset and depth
        self.conv_reg_prevs = nn.ModuleList()
        self.conv_regs = nn.ModuleList()
        for i in range(2):
            reg_dim = self.group_reg_dims[i]
            reg_branch_channels = self.reg_branch[i]
            out_channel = self.out_channels[i]
            if len(reg_branch_channels) > 0:
                self.conv_reg_prevs.append(
                    self._init_branch(
                        conv_channels=reg_branch_channels,
                        conv_strides=(1,) * len(reg_branch_channels)))
                self.conv_regs.append(nn.Conv2d(out_channel, reg_dim, 1))
            else:
                self.conv_reg_prevs.append(None)
                self.conv_regs.append(
                    nn.Conv2d(self.feat_channels, reg_dim, 1))
        # For uvd and sigma
        self.conv_pose_prevs = nn.ModuleList()
        self.conv_poses = nn.ModuleList()
        for i in range(2, 4):  # offset and sigma
            reg_dim = self.group_reg_dims[i]
            reg_branch_channels = self.reg_branch[i]
            out_channel = self.out_channels[i]
            if len(reg_branch_channels) > 0:
                self.conv_pose_prevs.append(
                    self._init_branch(
                        conv_channels=reg_branch_channels,
                        conv_strides=(1,) * len(reg_branch_channels)))
                self.conv_poses.append(nn.Conv2d(out_channel, reg_dim, 1))
            else:
                self.conv_pose_prevs.append(None)
                self.conv_poses.append(
                    nn.Conv2d(self.feat_channels, reg_dim, 1))

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_pose_convs()
        super()._init_layers()
        self.conv_centerness_prev = self._init_branch(
            conv_channels=self.centerness_branch,
            conv_strides=(1,) * len(self.centerness_branch))
        self.conv_centerness = nn.Conv2d(self.centerness_branch[-1], 1, 1)
        self.scales = nn.ModuleList([
            nn.ModuleList([Scale(1.0) for _ in self.group_reg_dims]) for _ in self.strides
        ])  # only for offset, depth, uvd and sigma
        self.biases = nn.ModuleList([Bias(0.0, use_bias=False) for _ in self.strides])  # for depth

    def forward(self, feats):
        output = multi_apply(self.forward_single, feats, self.scales, self.biases, self.strides)
        return output

    @force_fp32(apply_to=('x',), out_fp16=True)
    def _forward_single(self, x):
        cls_feat = x
        reg_feat = x
        pose_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        # clone the cls_feat for reusing the feature map afterwards
        clone_cls_feat = cls_feat.clone()
        for conv_cls_prev_layer in self.conv_cls_prev:
            clone_cls_feat = conv_cls_prev_layer(clone_cls_feat)
        cls_score = self.conv_cls(clone_cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        for pose_layer in self.pose_convs:
            pose_feat = pose_layer(pose_feat)
        pose_pred = []
        for i in range(len(self.group_reg_dims)):
            # clone the reg_feat for reusing the feature map afterwards
            if i < 2:
                clone_reg_feat = reg_feat.clone()
                conv_reg_prevs = self.conv_reg_prevs[i]
                conv_regs = self.conv_regs[i]
            else:
                clone_reg_feat = pose_feat.clone()
                conv_reg_prevs = self.conv_pose_prevs[i - 2]
                conv_regs = self.conv_poses[i - 2]

            if len(self.reg_branch[i]) > 0:
                for conv_reg_prev_layer in conv_reg_prevs:
                    clone_reg_feat = conv_reg_prev_layer(clone_reg_feat)
            pose_pred.append(conv_regs(clone_reg_feat))
        pose_pred = torch.cat(pose_pred, dim=1)

        return cls_score, pose_pred, cls_feat, reg_feat, pose_feat

    @force_fp32(apply_to=('cls_feat', 'reg_feat'), out_fp16=True)
    def _forward_centerness(self, cls_feat, reg_feat):
        if self.centerness_on_reg:
            clone_reg_feat = reg_feat.clone()
            for conv_centerness_prev_layer in self.conv_centerness_prev:
                clone_reg_feat = conv_centerness_prev_layer(clone_reg_feat)
            centerness = self.conv_centerness(clone_reg_feat)
        else:
            clone_cls_feat = cls_feat.clone()
            for conv_centerness_prev_layer in self.conv_centerness_prev:
                clone_cls_feat = conv_centerness_prev_layer(clone_cls_feat)
            centerness = self.conv_centerness(clone_cls_feat)
        return centerness

    def forward_single(self, x, scale, bias, stride):
        cls_score, pose_pred, cls_feat, reg_feat, pose_feat = self._forward_single(x)

        centerness = self._forward_centerness(cls_feat, reg_feat)

        scale_offset, scale_depth, scale_uv, scale_d = scale[0:4]
        clone_pose_pred = pose_pred.clone()
        pose_pred[:, :2] = scale_offset(clone_pose_pred[:, :2]).float()
        pose_pred[:, 2] = scale_depth(clone_pose_pred[:, 2]).float()
        clone_uvd = clone_pose_pred[:, 3:3 + self.num_joints * 3]
        uvd = pose_pred[:, 3:3 + self.num_joints * 3]
        uvd[:, 0::3] = scale_uv(clone_uvd[:, 0::3])
        uvd[:, 1::3] = scale_uv(clone_uvd[:, 1::3])
        uvd[:, 2::3] = scale_d(clone_uvd[:, 2::3])
        pose_pred[:, 2] = bias(pose_pred[:, 2])

        # relative root depth is 0
        pose_pred[:, 3 + self.root_idx * 3 + 2] = 0
        pose_pred[:, 3 + self.num_joints * 3 + self.root_idx * 3 + 2] = 1

        ref_uvd, _ = self.recursive_update_branch(
            pose_feat, pose_pred[:, 3:3 + self.num_joints * 3].clone())
        ref_uvd[:, self.root_idx * 3 + 2] = 0

        if not self.training:
            pose_pred[:, 3:3 + self.num_joints * 3] = ref_uvd
            pose_pred[:, 2] /= self.depth_factor
            pose_pred[:, 3 + self.root_idx * 3 + 2] = 0
            pose_pred[:, 3:3 + self.num_joints * 3:3] *= stride
            pose_pred[:, 4:3 + self.num_joints * 3:3] *= stride
            pose_pred[:, 5:3 + self.num_joints * 3:3] *= self.z_norm

        if self.training:
            return cls_score, pose_pred, centerness, ref_uvd
        else:
            return cls_score, pose_pred, centerness

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    @force_fp32(
        apply_to=('cls_scores', 'pose_preds', 'centernesses', 'aux_pose_preds'))
    def loss(self,
             cls_scores,
             pose_preds,
             centernesses,
             aux_pose_preds,
             gt_bboxes,
             gt_labels,
             gt_poses_3d,
             gt_labels_3d,
             centers2d,
             depths,
             img_metas,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(pose_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, pose_preds[0].dtype, pose_preds[0].device)
        labels_3d, pose_targets_3d, centerness_targets = \
            self.get_targets(
                all_level_points, gt_bboxes, gt_labels, gt_poses_3d,
                gt_labels_3d, centers2d, depths)

        num_imgs = cls_scores[0].size(0)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_pose_preds = [
            pose_pred.permute(0, 2, 3, 1).reshape(-1, sum(self.group_reg_dims))
            for pose_pred in pose_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_strides = [
            x.new_zeros(x.shape).fill_(self.strides[i]) for i, x in enumerate(flatten_centerness)
        ]
        flatten_aux_pose_pred = [
            p.permute(0, 2, 3, 1).reshape(-1, self.num_joints * 3)
            for p in aux_pose_preds
        ]

        flatten_strides = torch.cat(flatten_strides)
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_pose_preds = torch.cat(flatten_pose_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels_3d = torch.cat(labels_3d)
        flatten_pose_targets_3d = torch.cat(pose_targets_3d)
        flatten_centerness_targets = torch.cat(centerness_targets)
        flatten_aux_pose_pred = torch.cat(flatten_aux_pose_pred)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels_3d >= 0)
                    & (flatten_labels_3d < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(
            flatten_cls_scores,
            flatten_labels_3d,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_pose_preds = flatten_pose_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_strides = flatten_strides[pos_inds]

        if num_pos > 0:
            pos_pose_targets_3d = flatten_pose_targets_3d[pos_inds]
            pos_centerness_targets = flatten_centerness_targets[pos_inds]
            pos_aux_pose_pred = flatten_aux_pose_pred[pos_inds]

            pose_weights = pos_centerness_targets.new_ones(
                len(pos_centerness_targets), sum(self.group_reg_dims))
            equal_weights = pos_centerness_targets.new_ones(
                pos_centerness_targets.shape)

            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                assert len(code_weight) == sum(self.group_reg_dims)
                pose_weights = pose_weights * pose_weights.new_tensor(
                    code_weight)

            gt_uvd = pos_pose_targets_3d[:, 3:3 + self.num_joints * 3]
            is_2d = (gt_uvd[:, 2::3] == 0).all(dim=1)
            is_3d = ~is_2d
            if is_3d.sum() > 0:
                # s = pos_pose_preds[is_3d, 2].sigmoid() + 1e-9
                # loss_depth = torch.log(s) + \
                #              torch.abs(pos_pose_preds[is_3d, 2] - pos_pose_targets_3d[is_3d, 2] * self.depth_factor) / \
                #              np.sqrt(2) * s
                # loss_depth = (loss_depth * pose_weights[is_3d, 2]).sum() / equal_weights[is_3d].sum()
                loss_depth = self.loss_reg(
                    pos_pose_preds[is_3d, 2],
                    pos_pose_targets_3d[is_3d, 2] * self.depth_factor,
                    weight=pose_weights[is_3d, 2],
                    avg_factor=equal_weights[is_3d].sum())
            else:
                loss_depth = pos_pose_preds[0, 2] - pos_pose_preds[0, 2]

            uvd = pos_pose_preds[:, 3:3 + self.num_joints * 3].clone()
            uvd_update = pos_aux_pose_pred.view(num_pos, self.num_joints, 3)
            sigma = pos_pose_preds[:, 3 + self.num_joints * 3:].clone()

            # 2D annotations have no depth information
            uvd[is_2d, 2::3] = 0
            uvd_update[is_2d, :, 2] = 0
            sigma[is_2d, 2::3] = 1

            # gt_uvd:root-to-joint -> real_gt_uvd:pixel-to-joint
            diff = pos_pose_targets_3d[:, :3] * pos_strides[:, None]
            diff[:, 2] = 0
            diff = diff[:, None].expand(num_pos, self.num_joints, 3).reshape(num_pos, -1)
            gt_uvd = pos_pose_targets_3d[:, 3:3 + self.num_joints * 3]
            gt_uvd_weight = pos_pose_targets_3d[:, 3 + self.num_joints * 3:]
            real_gt_uvd = gt_uvd - diff

            real_gt_uvd = real_gt_uvd.reshape(-1, self.num_joints, 3)
            uvd = uvd.reshape(-1, self.num_joints, 3)
            gt_uvd_weight = gt_uvd_weight.reshape(-1, self.num_joints, 1).expand_as(real_gt_uvd)

            # normalize xy and depth
            real_gt_uvd[..., :2] = real_gt_uvd[..., :2] / pos_strides[:, None, None]
            real_gt_uvd[..., 2] = real_gt_uvd[..., 2] / self.z_norm

            # rle loss for joint offset
            sigma = sigma.sigmoid().reshape(-1, self.num_joints, 3) + 1e-9
            if self.recursive_update.get('prev_loss', False):
                uvd = torch.cat([uvd_update, uvd], dim=1)
                real_gt_uvd = real_gt_uvd.tile(1, 2, 1)
                sigma = sigma.tile(1, 2, 1)
                gt_uvd_weight = gt_uvd_weight.tile(1, 2, 1)

                bar_mu = (uvd - real_gt_uvd) / sigma
                is_2d = (real_gt_uvd[..., 2] == 0).all(dim=1)
                bar_mu_2d = bar_mu[is_2d][..., :2]
                bar_mu_3d = bar_mu[~is_2d]

                log_phi = bar_mu.new_zeros(*bar_mu.shape[:2], 1)
                if is_2d.any():
                    bar_mu_2d_update, bar_mu_2d = torch.split(bar_mu_2d, [self.num_joints, self.num_joints], dim=1)
                    log_phi_2d_update = self.flow2d_update.log_prob(
                        bar_mu_2d_update.reshape(-1, 2))  # [num_pos*num_joints]
                    log_phi_2d = self.flow2d.log_prob(bar_mu_2d.reshape(-1, 2))  # [num_pos*num_joints]
                    log_phi_2d = torch.cat([log_phi_2d_update.view(-1, self.num_joints),
                                            log_phi_2d.view(-1, self.num_joints)], dim=1)

                    log_phi[is_2d] = log_phi_2d.view(-1, bar_mu.size(1), 1)
                if (~is_2d).any():
                    bar_mu_3d_update, bar_mu_3d = torch.split(bar_mu_3d, [self.num_joints, self.num_joints], dim=1)
                    log_phi_3d_update = self.flow3d_update.log_prob(
                        bar_mu_3d_update.reshape(-1, 3))  # [num_pos*num_joints]
                    log_phi_3d = self.flow3d.log_prob(bar_mu_3d.reshape(-1, 3))  # [num_pos*num_joints]
                    log_phi_3d = torch.cat([log_phi_3d_update.view(-1, self.num_joints),
                                            log_phi_3d.view(-1, self.num_joints)], dim=1)

                    log_phi[~is_2d] = log_phi_3d.view(-1, bar_mu.size(1), 1)

            else:
                uvd = uvd_update

                bar_mu = (uvd - real_gt_uvd) / sigma    # [num_pos, num_joints, 3]
                is_2d = (real_gt_uvd[..., 2] == 0).all(dim=1)
                bar_mu_2d = bar_mu[is_2d].reshape(-1, 3)[:, :2]
                bar_mu_3d = bar_mu[~is_2d].reshape(-1, 3)

                log_phi = bar_mu.new_zeros(*bar_mu.shape[:2], 1)
                if is_2d.any():
                    log_phi_2d = self.flow2d.log_prob(bar_mu_2d)
                    log_phi[is_2d] = log_phi_2d.view(-1, bar_mu.size(1), 1)
                if (~is_2d).any():
                    log_phi_3d = self.flow3d.log_prob(bar_mu_3d)
                    log_phi[~is_2d] = log_phi_3d.view(-1, bar_mu.size(1), 1)

            nf_loss = torch.log(sigma) - log_phi

            loss_pose = self.loss_pose(
                nf_loss,
                uvd,
                sigma,
                real_gt_uvd,
                gt_uvd_weight,
                weight=pose_weights[:, None, [3]],
                avg_factor=equal_weights.sum())

            loss_pose = loss_pose

            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)

        else:
            loss_zero = (flatten_cls_scores[0, 0] - flatten_cls_scores[0, 0]).clone()
            loss_cls = loss_zero
            loss_depth = loss_zero
            loss_pose = loss_zero
            loss_centerness = loss_zero

        loss_dict = dict(
            loss_cls=loss_cls,
            loss_depth=loss_depth,
            loss_pose=loss_pose,
            loss_centerness=loss_centerness)

        return loss_dict

    def get_targets(self, points, gt_bboxes_list, gt_labels_list,
                    gt_poses_3d_list, gt_labels_3d_list, centers2d_list,
                    depths_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get targets of each image
        _, _, labels_3d_list, pose_targets_3d_list, centerness_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            gt_poses_3d_list,
            gt_labels_3d_list,
            centers2d_list,
            depths_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points,
        )

        # split to per img, per level
        labels_3d_list = [
            labels_3d.split(num_points, 0) for labels_3d in labels_3d_list
        ]
        pose_targets_3d_list = [
            pose_targets_3d.split(num_points, 0)
            for pose_targets_3d in pose_targets_3d_list
        ]
        centerness_targets_list = [
            centerness_targets.split(num_points, 0)
            for centerness_targets in centerness_targets_list
        ]

        # concat per level image
        concat_lvl_labels_3d = []
        concat_lvl_pose_targets_3d = []
        concat_lvl_centerness_targets = []

        for i in range(num_levels):
            concat_lvl_labels_3d.append(
                torch.cat([labels[i] for labels in labels_3d_list]))
            concat_lvl_centerness_targets.append(
                torch.cat([centerness_targets[i] for centerness_targets in centerness_targets_list
                           ]))
            pose_targets_3d = torch.cat([
                pose_targets_3d[i] for pose_targets_3d in pose_targets_3d_list
            ])
            # normalize by stride
            pose_targets_3d[:, :2] = pose_targets_3d[:, :2] / self.strides[i]
            concat_lvl_pose_targets_3d.append(pose_targets_3d)
        return concat_lvl_labels_3d, concat_lvl_pose_targets_3d, concat_lvl_centerness_targets

    def _get_target_single(self, gt_bboxes, gt_labels, gt_poses_3d,
                           gt_labels_3d, centers2d, depths,
                           points, regress_ranges, num_points_per_lvl,
                           ):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if not isinstance(gt_poses_3d, torch.Tensor):
            gt_poses_3d = gt_poses_3d.tensor.to(gt_bboxes.device)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_labels_3d.new_full(
                       (num_points,), self.background_label), \
                   gt_poses_3d.new_zeros((num_points, 3 + 4 * self.num_joints)), \
                   gt_poses_3d.new_zeros((num_points,))

        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        centers2d = centers2d[None].expand(num_points, num_gts, 2)
        gt_poses_3d = gt_poses_3d[None].expand(num_points, num_gts, 3 + 4 * self.num_joints)
        depths = depths[None, :, None].expand(num_points, num_gts, 1)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        # dx, dy to root joint
        delta_xs = (xs - centers2d[..., 0])[..., None]
        delta_ys = (ys - centers2d[..., 1])[..., None]

        uvds = gt_poses_3d[..., 3:3 + self.num_joints * 3].reshape(num_points, num_gts, self.num_joints, 3)
        # relative uvd, s.t., point + gt(points_to_roots) + gt(duvd) == gt(uvd)
        duvds = uvds - gt_poses_3d[..., None, :3]
        duvds[..., 2] = uvds[..., 2]
        duvds = duvds.view(num_points, num_gts, self.num_joints * 3)
        visible = gt_poses_3d[..., 3 + self.num_joints * 3:]

        # pose_targets_3d: [num_points, num_gts, (dx-root, dy-root, depth-root, duvd, visible)]
        pose_targets_3d = torch.cat((delta_xs, delta_ys, depths, duvds, visible), dim=-1)

        # only used for assign targets by max_regress_distance
        gt_offset = duvds.view(num_points, num_gts, self.num_joints, 3)[..., :2]
        gt_vis = visible
        gt_offset_len = torch.sqrt(torch.pow(gt_offset, 2).sum(-1)) * gt_vis
        reg_targets = gt_offset_len

        # center sampling following FCOS3D
        # condition1: inside a `center bbox`
        radius = self.center_sample_radius
        center_xs = centers2d[..., 0]
        center_ys = centers2d[..., 1]
        center_gts = torch.zeros_like(gt_bboxes)
        stride = center_xs.new_zeros(center_xs.shape)

        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end

        center_gts[..., 0] = center_xs - stride
        center_gts[..., 1] = center_ys - stride
        center_gts[..., 2] = center_xs + stride
        center_gts[..., 3] = center_ys + stride

        # assign targets for pixels around root centers
        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1]))

        # center-based criterion to deal with ambiguity
        dists = torch.sqrt(torch.sum(pose_targets_3d[..., :2] ** 2, dim=-1))  # [num_points, num_gts]
        dists[inside_gt_bbox_mask == 0] = INF
        dists[inside_regress_range == 0] = INF
        min_dist, min_dist_inds = dists.min(dim=1)

        labels = gt_labels[min_dist_inds]
        labels_3d = gt_labels_3d[min_dist_inds]
        labels[min_dist == INF] = self.background_label  # set as BG
        labels_3d[min_dist == INF] = self.background_label  # set as BG

        reg_targets = reg_targets[range(num_points), min_dist_inds]
        pose_targets_3d = pose_targets_3d[range(num_points), min_dist_inds]
        relative_dists = torch.sqrt(
            torch.sum(pose_targets_3d[..., :2] ** 2,
                      dim=-1)) / (1.414 * stride[:, 0])
        # [N, 1] / [N, 1]
        centerness_targets = torch.exp(-self.centerness_alpha * relative_dists)

        return labels, reg_targets, labels_3d, pose_targets_3d, centerness_targets

    def get_poses(self,
                  cls_scores,
                  pose_preds,
                  centernesses,
                  img_metas,
                  cfg=None,
                  rescale=None):
        assert len(cls_scores) == len(pose_preds) == len(centernesses)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, pose_preds[0].dtype, pose_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                pose_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            input_meta = img_metas[img_id]
            scores, poses, vis, centers = self._get_poses_single(
                cls_score_list, bbox_pred_list, centerness_pred_list, mlvl_points, input_meta,
                cfg, rescale)
            result_list.append({
                'poses': poses,
                'vis': vis,
                'centers': centers,
                'image_paths': [img_metas[img_id]['filename']],
                # for coco
                'scores': scores[..., 0].cpu().numpy().tolist(),
            })
        return result_list

    def _get_poses_single(self,
                          cls_scores,
                          pose_preds,
                          centernesses,
                          mlvl_points,
                          input_meta,
                          cfg,
                          rescale=False):
        scale_factor = input_meta['scale_factor']

        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(pose_preds) == len(mlvl_points)
        mlvl_centers2d = []
        mlvl_poses = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_vis = []

        for scores, pose_pred, centerness, points in \
                zip(cls_scores, pose_preds, centernesses, mlvl_points):
            if not self.training:
                assert scores.size()[-2:] == pose_pred.size()[-2:]
                scores = scores.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
                centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
                pose_pred = pose_pred.permute(1, 2, 0).reshape(-1, sum(self.group_reg_dims))

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                pose_pred = pose_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]

            pose_pred[:, :2] = points - pose_pred[:, :2]
            pose_center2d = pose_pred[:, :3].clone()

            joints3d_img = pose_pred[:, 3:3 + self.num_joints * 3]
            joints3d_img = joints3d_img.reshape(-1, self.num_joints, 3)
            joints3d_vis = pose_pred[:, 3 + self.num_joints * 3:3 + self.num_joints * 4]

            joints3d_vis[:] = 1

            rooots3d_img = pose_center2d[:, None].clone()
            rooots3d_img[:, 0, :2] = points
            scale = joints3d_img.new_tensor(scale_factor[:2])

            rooots3d_img[..., 2] *= torch.sqrt(scale.prod())
            pose_center2d[..., 2] *= torch.sqrt(scale.prod())

            joints3d_img = joints3d_img + rooots3d_img
            joints3d_img[..., :2] = joints3d_img[..., :2] / scale
            pose_center2d[:, :2] = pose_center2d[:, :2] / scale

            mlvl_centers2d.append(pose_center2d)
            mlvl_poses.append(joints3d_img)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_vis.append(joints3d_vis)

        mlvl_centers2d = torch.cat(mlvl_centers2d)
        mlvl_poses = torch.cat(mlvl_poses)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        mlvl_vis = torch.cat(mlvl_vis)

        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]
        score_thr = cfg.get('score_thr', 0.)
        if score_thr > 0:
            valid = mlvl_nms_scores[:, 0] > score_thr
            mlvl_nms_scores = mlvl_nms_scores[valid]
            mlvl_poses = mlvl_poses[valid]
            mlvl_centers2d = mlvl_centers2d[valid]
            mlvl_vis = mlvl_vis[valid]
        nms_post = cfg.get('nms_post', -1)

        if nms_post > 0 and len(mlvl_nms_scores) > 0:
            mlvl_xmaxymax = mlvl_poses[..., :2].max(1)[0]
            mlvl_xminymin = mlvl_poses[..., :2].min(1)[0]
            mlvl_areas = (mlvl_xmaxymax - mlvl_xminymin).prod(-1)
            to_nms = []
            for i in range(len(mlvl_poses)):
                to_nms.append({
                    'score': mlvl_nms_scores[i, 0].cpu().numpy(),
                    'keypoints': torch.cat([mlvl_poses[i, :, :2], mlvl_vis[i, :, None]], -1).cpu().numpy(),
                    'area': mlvl_areas[i].cpu().numpy(),
                })
            nms_thr = cfg.get('nms_thr', 0.9)
            nms_type = cfg.get('nms_type', 'hard')
            nms_post = cfg.get('nms_post', 100)
            if nms_type == 'hard':
                keep = oks_nms(to_nms, nms_thr).tolist()
                keep = keep[:nms_post]
            else:
                keep = soft_oks_nms(to_nms, nms_thr, max_dets=nms_post).tolist()
            mlvl_nms_scores = mlvl_nms_scores[keep]
            mlvl_poses = mlvl_poses[keep]
            mlvl_centers2d = mlvl_centers2d[keep]
            mlvl_vis = mlvl_vis[keep]

        return mlvl_nms_scores, mlvl_poses, mlvl_vis, mlvl_centers2d
