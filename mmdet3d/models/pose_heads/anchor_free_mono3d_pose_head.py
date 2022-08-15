import torch
from abc import abstractmethod
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from torch import nn as nn

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from .base_mono3d_dense_pose_head import BaseMono3DDensePoseHead


@HEADS.register_module()
class AnchorFreeMono3DPoseHead(BaseMono3DDensePoseHead):

    _version = 1

    def __init__(
            self,
            num_classes,
            in_channels,
            feat_channels=256,
            stacked_convs=4,
            strides=(4, 8, 16, 32, 64),
            dcn_on_last_conv=False,
            conv_bias='auto',
            background_label=None,
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
            num_joints=15,  # For mpii
            cls_branch=(128, 64),
            reg_branch=(
                (128, 64),  # to-root x-y offset
                (128, 64),  # to-root depth
                (128, 64),  # uvd
                (128, 64),  # sigma
            ),
            conv_cfg=None,
            norm_cfg=None,
            train_cfg=None,
            test_cfg=None,
            init_cfg=None):
        super(AnchorFreeMono3DPoseHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)
        self.loss_pose = build_loss(loss_pose)
        self.num_joints = num_joints
        self.group_reg_dims = [
            2,
            1,
            num_joints * 3,     # uvd
            num_joints * 3,     # sigma
        ]
        self.cls_branch = cls_branch
        self.reg_branch = reg_branch
        assert len(reg_branch) == len(self.group_reg_dims), 'The number of '\
            'element in reg_branch and group_reg_dims should be the same.'
        self.out_channels = []
        for reg_branch_channels in reg_branch:
            if len(reg_branch_channels) > 0:
                self.out_channels.append(reg_branch_channels[-1])
            else:
                self.out_channels.append(-1)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.background_label = (
            num_classes if background_label is None else background_label)
        # background_label should be either 0 or num_classes
        assert (self.background_label == 0
                or self.background_label == num_classes)

        self._init_layers()
        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal', name='conv_cls', std=0.01, bias_prob=0.01))

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_branch(self, conv_channels=(64), conv_strides=(1)):
        """Initialize conv layers as a prediction branch."""
        conv_before_pred = nn.ModuleList()
        if isinstance(conv_channels, int):
            conv_channels = [self.feat_channels] + [conv_channels]
            conv_strides = [conv_strides]
        else:
            conv_channels = [self.feat_channels] + list(conv_channels)
            conv_strides = list(conv_strides)
        for i in range(len(conv_strides)):
            conv_before_pred.append(
                ConvModule(
                    conv_channels[i],
                    conv_channels[i + 1],
                    3,
                    stride=conv_strides[i],
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        return conv_before_pred

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls_prev = self._init_branch(
            conv_channels=self.cls_branch,
            conv_strides=(1, ) * len(self.cls_branch))
        self.conv_cls = nn.Conv2d(self.cls_branch[-1], self.cls_out_channels,
                                  1)
        self.conv_reg_prevs = nn.ModuleList()
        self.conv_regs = nn.ModuleList()
        for i in range(len(self.group_reg_dims)):
            reg_dim = self.group_reg_dims[i]
            reg_branch_channels = self.reg_branch[i]
            out_channel = self.out_channels[i]
            if len(reg_branch_channels) > 0:
                self.conv_reg_prevs.append(
                    self._init_branch(
                        conv_channels=reg_branch_channels,
                        conv_strides=(1, ) * len(reg_branch_channels)))
                self.conv_regs.append(nn.Conv2d(out_channel, reg_dim, 1))
            else:
                self.conv_reg_prevs.append(None)
                self.conv_regs.append(
                    nn.Conv2d(self.feat_channels, reg_dim, 1))

    def init_weights(self):
        super().init_weights()

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually contain classification scores, bbox predictions, \
                and direction class predictions.
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                pose_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * bbox_code_size.
        """
        return multi_apply(self.forward_single, feats)[:5]

    def forward_single(self, x):
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.

        Returns:
            tuple: Scores for each class, bbox predictions, direction class,
                and attributes, features after classification and regression
                conv layers, some models needs these features like FCOS.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        # clone the cls_feat for reusing the feature map afterwards
        clone_cls_feat = cls_feat.clone()
        for conv_cls_prev_layer in self.conv_cls_prev:
            clone_cls_feat = conv_cls_prev_layer(clone_cls_feat)
        cls_score = self.conv_cls(clone_cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        pose_pred = []
        for i in range(len(self.group_reg_dims)):
            # clone the reg_feat for reusing the feature map afterwards
            clone_reg_feat = reg_feat.clone()
            if len(self.reg_branch[i]) > 0:
                for conv_reg_prev_layer in self.conv_reg_prevs[i]:
                    clone_reg_feat = conv_reg_prev_layer(clone_reg_feat)
            pose_pred.append(self.conv_regs[i](clone_reg_feat))
        pose_pred = torch.cat(pose_pred, dim=1)

        return cls_score, pose_pred, cls_feat, reg_feat

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points of a single scale level."""
        h, w = featmap_size
        x_range = torch.arange(w, dtype=dtype, device=device)
        y_range = torch.arange(h, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        return y, x

    def get_points(self, featmap_sizes, dtype, device, flatten=False):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(featmap_sizes[i], self.strides[i],
                                        dtype, device, flatten))
        return mlvl_points