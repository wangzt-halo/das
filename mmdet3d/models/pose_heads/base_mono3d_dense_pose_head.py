from abc import ABCMeta, abstractmethod
from mmcv.runner import BaseModule


class BaseMono3DDensePoseHead(BaseModule, metaclass=ABCMeta):
    """Base class for Monocular 3D DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseMono3DDensePoseHead, self).__init__(init_cfg=init_cfg)

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_poses(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_poses_3d=None,
                      gt_labels_3d=None,
                      centers2d=None,
                      depths=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_poses_3d,
                              gt_labels_3d, centers2d, depths, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
