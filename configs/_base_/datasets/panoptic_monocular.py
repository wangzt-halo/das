dataset_type = 'CMUPanopticDataset'
data_root = 'data/panoptic/'
class_names = [
    'person'
]
num_joints = 15
use_bbox_center = False
abs_dz = True

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotationsPose3D', with_bbox=True, with_label=True),
    dict(type='ResizePose',
         scale_depth=True,
         abs_dz=abs_dz,
         img_scale=[(1333, 512), (1333, 640)], multiscale_mode='range', keep_ratio=True),
    dict(type='RandomFlipPose3D', flip_ratio_bev_horizontal=0.5,
         flip_pairs=[[3, 9], [4, 10], [5, 11], [6, 12], [7, 13], [8, 14]],
         num_joints=num_joints),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.7, 1.3),
        saturation_range=(0.7, 1.3),
        hue_delta=18),
    dict(
        type='GlobalRotScaleTransPose',
        scale_depth=True,
        abs_dz=abs_dz,
        rot_range=[-0.0, 0.0],
        scale_ratio_range=[0.6, 1.4],
        translation_std=[0.15, 0.15],
        num_joints=15,
        img_norm_cfg=img_norm_cfg,
        use_bbox_center=use_bbox_center,
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundlePose3D', class_names=['person']),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_poses_3d',
            'gt_labels_3d', 'centers2d', 'depths'
        ],
        debug=False,
        num_joints=num_joints
    ),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsPose3D', with_pose_3d=True, with_label_3d=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlipPose3D', flip_ratio_bev_horizontal=0.0,
                 flip_pairs=[[3, 9], [4, 10], [5, 11], [6, 12], [7, 13], [8, 14]],
                 num_joints=num_joints),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundlePose3D', class_names=['person'], with_label=False),
            dict(type='Collect3D', keys=['img', 'gt_poses_3d', 'depths']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        use_bbox_center=use_bbox_center,
        norm_depth=True,
        abs_dz=abs_dz,
        depth_factor=1,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        use_bbox_center=use_bbox_center,
        norm_depth=True,
        abs_dz=abs_dz,
        depth_factor=1,
        ann_file=data_root + 'annotations/pizza.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        use_bbox_center=use_bbox_center,
        norm_depth=True,
        abs_dz=abs_dz,
        depth_factor=1,
        ann_file=data_root + 'annotations/pizza.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
)

