from mmdet.models.builder import DETECTORS
from .single_stage_mono3d import SingleStageMono3DDetector


@DETECTORS.register_module()
class DAS(SingleStageMono3DDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DAS, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_poses_3d,
                      gt_labels_3d,
                      centers2d,
                      depths,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_poses_3d,
                                              gt_labels_3d, centers2d, depths,
                                              gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        result = self.bbox_head.get_poses(
            *outs, img_metas, rescale=rescale)
        return result

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        raise NotImplementedError