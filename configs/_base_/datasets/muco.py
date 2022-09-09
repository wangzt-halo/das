class_names = [
    'person'
]

num_joints = 21
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline_muco = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotationsPose3D', with_bbox=True, with_label=True),
    dict(type='ResizePose',
         scale_depth=True,
         abs_dz=abs_dz,
         img_scale=[(1280, 512), (1280, 768)], multiscale_mode='range', keep_ratio=True),
    dict(type='RandomFlipPose3D', flip_ratio_bev_horizontal=0.5,
         flip_pairs=((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20)),
         num_joints=num_joints),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=14),
    dict(
        type='GlobalRotScaleTransPose',
        scale_depth=True,
        abs_dz=abs_dz,
        # rot_range=[-0.15, 0.15],
        rot_range=[-0.1, 0.1],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.15, 0.15],
        num_joints=num_joints,
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
        img_scale=(1280, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlipPose3D', flip_ratio_bev_horizontal=0.0,
                 flip_pairs=((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13)),
                 num_joints=17),
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
        type='MuCo3DHPDataset',
        data_root='data/muco/',
        yaml_file='data/muco/muco/train.yaml',
        ann_file='annotations/train.json',
        # img_prefix='data/muco/',
        classes=('person',),
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type='MuPots3DHP',
        data_root='data/mupots',
        ann_file='annotations/MuPoTS-3D.json',
        img_prefix='',
        classes=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type='MuPots3DHP',
        data_root='data/mupots',
        ann_file='annotations/MuPoTS-3D.json',
        img_prefix='',
        classes=class_names,
        pipeline=test_pipeline,
    ),
)
