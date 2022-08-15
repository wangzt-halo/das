import math
import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class RLELoss3D(nn.Module):
    ''' RLE Regression Loss 3D
    '''

    def __init__(self, residual=True, avg_factor=False, **kwargs):
        super(RLELoss3D, self).__init__()
        self.residual = residual
        self.avg_factor = avg_factor
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, nf_loss, uvd, sigma, gt_uvd, gt_uv_weight, weight=None, avg_factor=None):
        gt_uv_weight = gt_uv_weight.expand_as(gt_uvd)
        nf_loss = nf_loss * gt_uv_weight
        if gt_uv_weight[..., 0].sum() < 1:
            return gt_uv_weight[..., 0].sum()

        if self.residual:
            Q_logprob = self.logQ(gt_uvd, uvd, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob

        if weight is not None:
            loss = loss * weight

        if avg_factor is not None and self.avg_factor:
            return loss.sum() / avg_factor
        else:
            return loss.sum() / gt_uv_weight[..., 0].sum()

