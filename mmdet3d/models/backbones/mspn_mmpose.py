# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, MaxPool2d, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init, normal_init)
from mmcv.runner.checkpoint import _load_checkpoint, load_state_dict, load_checkpoint

from mmdet.utils import get_root_logger
from ..builder import BACKBONES
import copy


class _Bottleneck(nn.Module):
    """Bottleneck block for ResNet.
    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): ``"pytorch"`` or ``"caffe"``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        assert style in ['pytorch', 'caffe']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out



def get_state_dict(filename, map_location='cpu'):
    """Get state_dict from a file or URI.
    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``.
        map_location (str): Same as :func:`torch.load`.
    Returns:
        OrderedDict: The state_dict.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict_tmp = checkpoint['state_dict']
    else:
        state_dict_tmp = checkpoint

    state_dict = OrderedDict()
    # strip prefix of state_dict
    for k, v in state_dict_tmp.items():
        if k.startswith('module.backbone.'):
            state_dict[k[16:]] = v
        elif k.startswith('module.'):
            state_dict[k[7:]] = v
        elif k.startswith('backbone.'):
            state_dict[k[9:]] = v
        else:
            state_dict[k] = v

    return state_dict


class Bottleneck(_Bottleneck):
    expansion = 4
    """Bottleneck block for MSPN.
    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        stride (int): stride of the block. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels * 4, **kwargs)


class DownsampleModule(nn.Module):
    """Downsample module for MSPN.
    Args:
        block (nn.Module): Downsample block.
        num_blocks (list): Number of blocks in each downsample unit.
        num_units (int): Numbers of downsample units. Default: 4
        has_skip (bool): Have skip connections from prior upsample
            module or not. Default:False
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        in_channels (int): Number of channels of the input feature to
            downsample module. Default: 64
    """

    def __init__(self,
                 block,
                 num_blocks,
                 num_units=4,
                 has_skip=False,
                 norm_cfg=dict(type='BN'),
                 in_channels=64):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.has_skip = has_skip
        self.in_channels = in_channels
        assert len(num_blocks) == num_units
        self.num_blocks = num_blocks
        self.num_units = num_units
        self.norm_cfg = norm_cfg
        self.layer1 = self._make_layer(block, in_channels, num_blocks[0])
        for i in range(1, num_units):
            module_name = f'layer{i + 1}'
            self.add_module(
                module_name,
                self._make_layer(
                    block, in_channels * pow(2, i), num_blocks[i], stride=2))

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = ConvModule(
                self.in_channels,
                out_channels * block.expansion,
                kernel_size=1,
                stride=stride,
                padding=0,
                norm_cfg=self.norm_cfg,
                act_cfg=None,
                inplace=True)

        units = list()
        units.append(
            block(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                norm_cfg=self.norm_cfg))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            units.append(block(self.in_channels, out_channels))

        return nn.Sequential(*units)

    def forward(self, x, skip1, skip2):
        out = list()
        for i in range(self.num_units):
            module_name = f'layer{i + 1}'
            module_i = getattr(self, module_name)
            x = module_i(x)
            if self.has_skip:
                x = x + skip1[i] + skip2[i]
            out.append(x)
        out.reverse()

        return tuple(out)


class UpsampleUnit(nn.Module):
    """Upsample unit for upsample module.
    Args:
        ind (int): Indicates whether to interpolate (>0) and whether to
           generate feature map for the next hourglass-like module.
        num_units (int): Number of units that form a upsample module. Along
            with ind and gen_cross_conv, nm_units is used to decide whether
            to generate feature map for the next hourglass-like module.
        in_channels (int): Channel number of the skip-in feature maps from
            the corresponding downsample unit.
        unit_channels (int): Channel number in this unit. Default:256.
        gen_skip: (bool): Whether or not to generate skips for the posterior
            downsample module. Default:False
        gen_cross_conv (bool): Whether to generate feature map for the next
            hourglass-like module. Default:False
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        out_channels (in): Number of channels of feature output by upsample
            module. Must equal to in_channels of downsample module. Default:64
    """

    def __init__(self,
                 ind,
                 num_units,
                 in_channels,
                 unit_channels=256,
                 gen_skip=False,
                 gen_cross_conv=False,
                 norm_cfg=dict(type='BN'),
                 out_channels=64):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.num_units = num_units
        self.norm_cfg = norm_cfg
        self.in_skip = ConvModule(
            in_channels,
            unit_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=self.norm_cfg,
            act_cfg=None,
            inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self.ind = ind
        if self.ind > 0:
            self.up_conv = ConvModule(
                unit_channels,
                unit_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                act_cfg=None,
                inplace=True)

        self.gen_skip = gen_skip
        if self.gen_skip:
            self.out_skip1 = ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                inplace=True)

            self.out_skip2 = ConvModule(
                unit_channels,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                inplace=True)

        self.gen_cross_conv = gen_cross_conv
        if self.ind == num_units - 1 and self.gen_cross_conv:
            self.cross_conv = ConvModule(
                unit_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                inplace=True)

    def forward(self, x, up_x):
        out = self.in_skip(x)

        if self.ind > 0:
            up_x = F.interpolate(
                up_x,
                size=(x.size(2), x.size(3)),
                mode='bilinear',
                align_corners=True)
            up_x = self.up_conv(up_x)
            out = out + up_x
        out = self.relu(out)

        skip1 = None
        skip2 = None
        if self.gen_skip:
            skip1 = self.out_skip1(x)
            skip2 = self.out_skip2(out)

        cross_conv = None
        if self.ind == self.num_units - 1 and self.gen_cross_conv:
            cross_conv = self.cross_conv(out)

        return out, skip1, skip2, cross_conv


class UpsampleModule(nn.Module):
    """Upsample module for MSPN.
    Args:
        unit_channels (int): Channel number in the upsample units.
            Default:256.
        num_units (int): Numbers of upsample units. Default: 4
        gen_skip (bool): Whether to generate skip for posterior downsample
            module or not. Default:False
        gen_cross_conv (bool): Whether to generate feature map for the next
            hourglass-like module. Default:False
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        out_channels (int): Number of channels of feature output by upsample
            module. Must equal to in_channels of downsample module. Default:64
    """

    def __init__(self,
                 unit_channels=256,
                 num_units=4,
                 gen_skip=False,
                 gen_cross_conv=False,
                 norm_cfg=dict(type='BN'),
                 out_channels=64):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        # if norm_cfg['type'] != 'SyncBN':
        #     norm_cfg = dict(type='GN', num_groups=32,)
        super().__init__()
        self.in_channels = list()
        for i in range(num_units):
            self.in_channels.append(Bottleneck.expansion * out_channels *
                                    pow(2, i))
        self.in_channels.reverse()
        self.num_units = num_units
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.norm_cfg = norm_cfg
        for i in range(num_units):
            module_name = f'up{i + 1}'
            self.add_module(
                module_name,
                UpsampleUnit(
                    i,
                    self.num_units,
                    self.in_channels[i],
                    unit_channels,
                    self.gen_skip,
                    self.gen_cross_conv,
                    norm_cfg=self.norm_cfg,
                    out_channels=64))

    def forward(self, x):
        out = list()
        skip1 = list()
        skip2 = list()
        cross_conv = None
        for i in range(self.num_units):
            module_i = getattr(self, f'up{i + 1}')
            if i == 0:
                outi, skip1_i, skip2_i, _ = module_i(x[i], None)
            elif i == self.num_units - 1:
                outi, skip1_i, skip2_i, cross_conv = module_i(x[i], out[i - 1])
            else:
                outi, skip1_i, skip2_i, _ = module_i(x[i], out[i - 1])
            out.append(outi)
            skip1.append(skip1_i)
            skip2.append(skip2_i)
        skip1.reverse()
        skip2.reverse()

        return out, skip1, skip2, cross_conv


class SingleStageNetwork(nn.Module):
    """Single_stage Network.
    Args:
        unit_channels (int): Channel number in the upsample units. Default:256.
        num_units (int): Numbers of downsample/upsample units. Default: 4
        gen_skip (bool): Whether to generate skip for posterior downsample
            module or not. Default:False
        gen_cross_conv (bool): Whether to generate feature map for the next
            hourglass-like module. Default:False
        has_skip (bool): Have skip connections from prior upsample
            module or not. Default:False
        num_blocks (list): Number of blocks in each downsample unit.
            Default: [2, 2, 2, 2] Note: Make sure num_units==len(num_blocks)
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        in_channels (int): Number of channels of the feature from ResNetTop.
            Default: 64.
    """

    def __init__(self,
                 has_skip=False,
                 gen_skip=False,
                 gen_cross_conv=False,
                 unit_channels=256,
                 num_units=4,
                 num_blocks=[2, 2, 2, 2],
                 norm_cfg=dict(type='BN'),
                 in_channels=64):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        num_blocks = cp.deepcopy(num_blocks)
        super().__init__()
        assert len(num_blocks) == num_units
        self.has_skip = has_skip
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.num_units = num_units
        self.unit_channels = unit_channels
        self.num_blocks = num_blocks
        self.norm_cfg = norm_cfg

        self.downsample = DownsampleModule(Bottleneck, num_blocks, num_units,
                                           has_skip, norm_cfg, in_channels)
        self.upsample = UpsampleModule(unit_channels, num_units, gen_skip,
                                       gen_cross_conv, norm_cfg, in_channels)

    def forward(self, x, skip1, skip2):
        mid = self.downsample(x, skip1, skip2)
        out, skip1, skip2, cross_conv = self.upsample(mid)

        return out, skip1, skip2, cross_conv


class ResNetTop(nn.Module):
    """ResNet top for MSPN.
    Args:
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        channels (int): Number of channels of the feature output by ResNetTop.
    """

    def __init__(self, norm_cfg=dict(type='BN'), channels=64):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.top = nn.Sequential(
            ConvModule(
                3,
                channels,
                kernel_size=7,
                stride=2,
                padding=3,
                norm_cfg=norm_cfg,
                inplace=True), MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, img):
        return self.top(img)


@BACKBONES.register_module()
class MSPN2(nn.Module):
    """MSPN backbone. Paper ref: Li et al. "Rethinking on Multi-Stage Networks
    for Human Pose Estimation" (CVPR 2020).
    Args:
        unit_channels (int): Number of Channels in an upsample unit.
            Default: 256
        num_stages (int): Number of stages in a multi-stage MSPN. Default: 4
        num_units (int): NUmber of downsample/upsample units in a single-stage
            network. Default: 4
            Note: Make sure num_units == len(self.num_blocks)
        num_blocks (list): Number of bottlenecks in each
            downsample unit. Default: [2, 2, 2, 2]
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        res_top_channels (int): Number of channels of feature from ResNetTop.
            Default: 64.
    Example:
        >>> from mmpose.models import MSPN
        >>> import torch
        >>> self = MSPN(num_stages=2,num_units=2,num_blocks=[2,2])
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 511, 511)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     for feature in level_output:
        ...         print(tuple(feature.shape))
        ...
        (1, 256, 64, 64)
        (1, 256, 128, 128)
        (1, 256, 64, 64)
        (1, 256, 128, 128)
    """

    def __init__(self,
                 unit_channels=256,
                 num_stages=4,
                 num_units=4,
                 num_blocks=[2, 2, 2, 2],
                 norm_cfg=dict(type='BN'),
                 res_top_channels=64,
                 frozen_stages=-1, norm_eval=False,
                 pretrained=None):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        num_blocks = cp.deepcopy(num_blocks)
        super().__init__()
        self.unit_channels = unit_channels
        self.num_stages = num_stages
        self.num_units = num_units
        self.num_blocks = num_blocks
        self.norm_cfg = norm_cfg

        assert self.num_stages > 0
        assert self.num_units > 1
        assert self.num_units == len(self.num_blocks)
        self.top = ResNetTop(norm_cfg=norm_cfg)
        self.multi_stage_mspn = nn.ModuleList([])
        for i in range(self.num_stages):
            if i == 0:
                has_skip = False
            else:
                has_skip = True
            if i != self.num_stages - 1:
                gen_skip = True
                gen_cross_conv = True
            else:
                gen_skip = False
                gen_cross_conv = False
            self.multi_stage_mspn.append(
                SingleStageNetwork(has_skip, gen_skip, gen_cross_conv,
                                   unit_channels, num_units, num_blocks,
                                   norm_cfg, res_top_channels))

        self.pretrained = pretrained

    def _frozen_stage(self, frozen_stage=-1):
        if frozen_stage >= 0:
            self.top.eval()
            for p in self.top.parameters():
                p.requires_grad = False
        if frozen_stage >= 1:
            resnet = self.multi_stage_mspn[0].downsample
            for i in range(1, frozen_stage + 1):
                layer = getattr(resnet, 'layer%d'%i)
                layer.eval()
                for p in layer.parameters():
                    p.requires_grad = False

    def _frozen_bn(self):
        resnet = [self.top, self.multi_stage_mspn[0].downsample]
        for r in resnet:
            for m in r.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                    m.eval()

    def forward(self, x):
        """Model forward function."""
        out_feats = []
        skip1 = None
        skip2 = None
        x = self.top(x)
        for i in range(self.num_stages):
            out, skip1, skip2, x = self.multi_stage_mspn[i](x, skip1, skip2)
            out_feats.append(out)

        return out_feats[-1][::-1]

    def init_weights(self, pretrained=None):
        pretrained = self.pretrained
        # import ipdb;ipdb.set_trace()
        """Initialize model weights."""
        if pretrained.startswith('weights/'):
            loaded = torch.load(pretrained, map_location='cpu')['state_dict']
            new_state_dict = {}
            for k, v in loaded.items():
                if k.startswith('backbone.'):
                    new_k = k.replace('backbone.', '')
                    new_state_dict[new_k] = v
            logger = get_root_logger()
            load_state_dict(self, new_state_dict, strict=False, logger=logger)
        else:
            for m in self.multi_stage_mspn.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)

            for m in self.top.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)

            if isinstance(pretrained, str):
                logger = get_root_logger()
                state_dict_tmp = get_state_dict(pretrained)
                state_dict = OrderedDict()
                state_dict['top'] = OrderedDict()
                state_dict['bottlenecks'] = OrderedDict()
                for k, v in state_dict_tmp.items():
                    if k.startswith('layer'):
                        if 'downsample.0' in k:
                            state_dict['bottlenecks'][k.replace(
                                'downsample.0', 'downsample.conv')] = v
                        elif 'downsample.1' in k:
                            state_dict['bottlenecks'][k.replace(
                                'downsample.1', 'downsample.bn')] = v
                        else:
                            state_dict['bottlenecks'][k] = v
                    elif k.startswith('conv1'):
                        state_dict['top'][k.replace('conv1', 'top.0.conv')] = v
                    elif k.startswith('bn1'):
                        state_dict['top'][k.replace('bn1', 'top.0.bn')] = v

                load_state_dict(
                    self.top, state_dict['top'], strict=False, logger=logger)
                for i in range(self.num_stages):
                    load_state_dict(
                        self.multi_stage_mspn[i].downsample,
                        state_dict['bottlenecks'],
                        strict=False,
                        logger=logger)