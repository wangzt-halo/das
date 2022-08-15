import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32


def offset_sample_core(offset, offset_shape, sampling_locations, diff, offset_conf):
    '''
    :param offset:                  (N:batch*num_joints*num_heads, dim, H, W)
    :param offset_shape:            (offset.shape)
    :param sampling_locations:      (N, H, W, 2), ranged from 0 to 1
    :param diff:                    (N, dim, H, W) or int
    :param offset_conf:             (batch*num_joints*num_heads, dim, H, W)
    :param sigma:                   (batch*num_joints*num_heads, dim, H, W)
    :return:
    '''
    n, dim, h, w = offset.shape
    n2, h2, w2, _ = sampling_locations.shape
    (batch, num_joints, num_heads, _, _, _) = offset_shape
    assert (n, h, w) == (n2, h2, w2), (offset.shape, sampling_locations.shape)
    sampling_grids = 2 * sampling_locations - 1
    feat = torch.cat([offset, offset_conf], dim=1)
    sampling_feat = F.grid_sample(feat.float(), sampling_grids, mode='bilinear', padding_mode='zeros', align_corners=False)
    sampling_offset, sampling_conf = torch.split(sampling_feat, [dim, dim], dim=1)
    sampling_offset_per_head = sampling_offset + diff  # (batch*num_joints*num_heads, dim, H, W)
    sampling_conf = sampling_conf.reshape(batch * num_joints, num_heads, dim, h, w).softmax(dim=1)
    sampling_offset = sampling_offset_per_head.reshape(batch * num_joints, num_heads, dim, h, w) * sampling_conf
    sampling_offset = sampling_offset.sum(1)  # (batch, num)
    return sampling_offset.view(batch, num_joints, dim, h, w), None


def offset_sample(uvd, sampling_offset, joint_conf, dim_info, points_locations):
    '''
    :param uvd:                 (batch, num_joints*dim, H, W)
    :param sampling_offset:     (batch, num_joints*num_heads*2, H, W)
    :param joint_conf:          (batch, num_joints*dim, H, W)
    :param dim_info:            tuple(batch, num_joints, num_heads, dim)
    :param points_locations:    (2, H, W)
    :return:
    '''

    batch, num_joints, num_heads, dim = dim_info
    h, w = uvd.shape[-2:]
    normalize_factor = uvd.new_tensor([w, h]).view(1, 2, 1, 1)

    # sample heads from target position
    uvd = uvd.view(batch*num_joints, dim, h, w)
    off_to_target = uvd[:, :2]
    target_loc = points_locations + off_to_target
    target_loc = (target_loc / normalize_factor).permute(0, 2, 3, 1)

    sampl_grids = 2 * target_loc - 1
    sampl_off_from_target = sampling_offset.view(batch*num_joints, num_heads*2, h, w)
    sampl_off_from_target = F.grid_sample(sampl_off_from_target.float(), sampl_grids,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)    # (batch*num_joints, num_heads*2, h, w)
    sampl_off_from_target = sampl_off_from_target.view(batch*num_joints, num_heads, 2, h, w)
    sampl_off_from_target = sampl_off_from_target + off_to_target[:, None]

    # sample heads
    sampl_off_from_source = sampling_offset.view(batch*num_joints, num_heads, 2, h, w)
    sampl_off = torch.cat([sampl_off_from_target, sampl_off_from_source], dim=1)    # (batch*num_joints, 2*num_heads, 2, h, w)
    sampl_off = sampl_off.view(batch*num_joints*2*num_heads, 2, h, w)
    sampl_loc = points_locations + sampl_off
    sampl_loc = (sampl_loc / normalize_factor).permute(0, 2, 3, 1)

    # joint confidence
    offset_conf = joint_conf.view(batch*num_joints, dim, h, w)
    offset_conf = offset_conf.repeat_interleave(2*num_heads, dim=0)

    # offset_sample
    offset_shape = (batch, num_joints, 2*num_heads, dim, h, w)
    offset = uvd.repeat_interleave(2*num_heads, dim=0)

    if dim == 3:
        diff = torch.cat([sampl_off, sampl_off.new_zeros([sampl_off.size(0), 1, h, w])], dim=1)
    else:
        diff = sampl_off
    new_uvd, new_sigma = offset_sample_core(offset, offset_shape, sampl_loc, diff, offset_conf)

    return new_uvd, new_sigma


class DepthSample(nn.Module):
    def __init__(self, num_heads, in_channels, feat_channels=None, norm_type='hw', root_centered=False):
        super(DepthSample, self).__init__()
        if feat_channels is not None:
            pass
        else:
            feat_channels = in_channels

        self.update_feat_conv = ConvModule(in_channels, feat_channels, 3, padding=1,
                                           conv_cfg=dict(type='DCNv2'), norm_cfg=dict(type='GN', num_groups=32))
        self.num_heads = num_heads
        self.sample_conv = nn.Conv2d(feat_channels, num_heads*2, 1, bias=False)
        self.conf_conv = nn.Conv2d(feat_channels, 1, 1, bias=False)
        self.norm_type = norm_type
        self.root_centered = root_centered
        if root_centered:
            assert norm_type == 'hw'

        normal_(self.sample_conv.weight.data, 0, 1e-2)

    def _get_points_single(self, featmap):
        """Get points of a single scale level."""
        h, w = featmap.shape[-2:]
        x_range = torch.arange(w, dtype=featmap.dtype, device=featmap.device)
        y_range = torch.arange(h, dtype=featmap.dtype, device=featmap.device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x, y), dim=0) + 0.5  # (2, H, W)
        return points

    @force_fp32(apply_to=('feat',), out_fp16=True)
    def _update_feat(self, feat):
        return self.update_feat_conv(feat)

    def forward(self, feat, init_depth, stride, root_offset=None):
        batch, _, h, w = feat.shape
        normalize_factor = feat.new_tensor([w, h]).view(1, 2, 1, 1)
        identity = feat
        feat = self._update_feat(feat)
        if feat.size(1) == identity.size(1):
            feat = feat + identity

        points_location = self._get_points_single(feat)
        sampl_off = self.sample_conv(feat)
        num_heads = self.num_heads

        if self.root_centered:
            num_heads += 1
            root_offset = root_offset.detach()
            root_sampl_loc = ((points_location + root_offset) / normalize_factor).permute(0, 2, 3, 1)
            root_sampl_grids = 2 * root_sampl_loc - 1
            sampl_offset_from_root = F.grid_sample(sampl_off, root_sampl_grids, mode='bilinear', padding_mode='zeros', align_corners=False)
            sampl_offset_from_root = torch.cat([sampl_offset_from_root, root_offset.new_zeros(root_offset.shape)], dim=1)
            sampl_off = sampl_offset_from_root + root_offset.tile(1, num_heads, 1, 1)

        sampl_off = sampl_off.view(batch * num_heads, 2, h, w)

        if self.norm_type == 'hw':
            sampl_off = sampl_off / normalize_factor
        elif self.norm_type == 'stride':
            sampl_off = sampl_off / stride
        else:
            raise NotImplementedError
        sampl_loc = (sampl_off + points_location / normalize_factor).permute(0, 2, 3, 1)

        sampl_conf = self.conf_conv(feat)

        feat = torch.cat([init_depth, sampl_conf], dim=1)
        feat = feat.repeat_interleave(num_heads, dim=0)

        sampl_grids = 2 * sampl_loc - 1
        sampl_feat = F.grid_sample(feat, sampl_grids, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampl_depth, sampl_conf = torch.split(sampl_feat, [1, 1], dim=1)
        sampl_conf = sampl_conf.reshape(batch, num_heads, h, w).softmax(dim=1)
        sampl_depth = sampl_depth.reshape(batch, num_heads, h, w)
        sampl_depth = (sampl_depth * sampl_conf).sum(dim=1, keepdim=True)

        return sampl_depth


class NextLevelOffset(nn.Module):
    def __init__(self, num_joints, num_heads, in_channels, dim=3, **kwargs):
        super(NextLevelOffset, self).__init__()
        self.num_joints = num_joints
        self.num_heads = num_heads
        self.dim = dim

        self.sampling_offset = nn.Conv2d(in_channels, num_joints * num_heads * 2, 1)
        self.sampling_conf = nn.Conv2d(in_channels, num_joints * dim, 1)

        normal_(self.sampling_offset.weight.data, 0, 1e-2)
        constant_(self.sampling_offset.bias.data, 0)

        self.update_feat_conv = ConvModule(in_channels, in_channels, 3, padding=1,
                                           conv_cfg=dict(type='DCNv2'), norm_cfg=dict(type='GN', num_groups=32))
        self.update_weight = nn.Conv2d(in_channels, num_joints * dim, 1)
        self.update_offset_value = nn.Conv2d(in_channels, num_joints * dim, 1)

    @force_fp32(apply_to=('feat',), out_fp16=True)
    def _update_feat(self, feat):
        return self.update_feat_conv(feat)

    def forward(self, feat, offset):

        feat = feat + self._update_feat(feat)

        sampling_offset = self.sampling_offset(feat)
        sampling_conf = self.sampling_conf(feat)

        offset_weight = self.update_weight(feat).sigmoid()
        next_offset = self.update_offset_value(feat)
        offset = (1 - offset_weight) * offset + offset_weight * next_offset

        return feat, offset, sampling_offset, sampling_conf


class RecursiveUpdateLayer(nn.Module):
    def __init__(self, num_joints, num_heads, in_channels, dim=3, **kwargs):
        super(RecursiveUpdateLayer, self).__init__()
        self.num_joints = num_joints
        self.num_heads = num_heads
        self.in_channels = in_channels
        assert dim in [2, 3]
        self.dim = dim

        self.next_level_offset = NextLevelOffset(num_joints, num_heads, in_channels, dim, **kwargs)

    def _get_points_single(self, featmap):
        """Get points of a single scale level."""
        h, w = featmap.shape[-2:]
        x_range = torch.arange(w, dtype=featmap.dtype, device=featmap.device)
        y_range = torch.arange(h, dtype=featmap.dtype, device=featmap.device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x, y), dim=0) + 0.5  # (2, H, W)
        return points

    def forward(self, feat, prev_offset):
        '''
        :param feat:                    (N, c, H, W)
        :param prev_offset:             (N, num_joints*dim, H, W)
        :return: new_feat,              (N, c, H, W)
                 new_offset,            (N, num_joints*dim, H, W)
        '''
        batch, _, h, w = feat.shape
        feat, prev_offset, sampling_offset, sampling_conf = self.next_level_offset(feat, prev_offset)

        points_location = self._get_points_single(feat)
        dim_info = (batch, self.num_joints, self.num_heads, self.dim)

        new_offset, _ = offset_sample(prev_offset, sampling_offset, sampling_conf, dim_info, points_location)

        return feat, new_offset.reshape(batch, self.num_joints * self.dim, h, w), None


class RecursiveUpdateBranch(nn.Module):
    def __init__(self, num_joints, num_heads, in_channels, feat_channels, num_layers=3, dim=3,
                 **kwargs):
        super(RecursiveUpdateBranch, self).__init__()
        self.num_layers = num_layers
        self.reduction = ConvModule(in_channels, feat_channels, 1,
                                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True))

        for i in range(num_layers):
            layer = RecursiveUpdateLayer(num_joints, num_heads, feat_channels, dim, **kwargs)
            self.add_module('layer_%d' % i, layer)

    def forward(self, feat, offset):
        feat = self.reduction(feat)
        for i in range(self.num_layers):
            layer = getattr(self, 'layer_%d' % i)
            feat, offset, _ = layer(feat, offset)
        return offset, None
