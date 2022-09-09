_base_ = [
    '../_base_/datasets/muco.py', '../_base_/models/das.py',
    '../_base_/schedules/mmdet_schedule_1x.py', '../_base_/default_runtime.py'
]

fpn_channels = 256
num_joints = 21
use_bbox_center = False
abs_dz = True
# model settings
model = dict(
    pretrained='weights/3xmspn50_coco_256x192-e348f18e_20201123.pth',
    backbone=dict(
        _delete_=True,
        type='MSPN2',
        unit_channels=256,
        num_stages=3,
        num_units=4,
        num_blocks=[3, 4, 6, 3],
        norm_cfg=dict(type='BN'),
        frozen_stages=1,
        norm_eval=False,
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 256, 256, 256],
        out_channels=fpn_channels,
        norm_cfg=dict(type='BN'),
        num_outs=4,
    ),
    bbox_head=dict(
        type='DASHead',
        stacked_convs=2,
        in_channels=fpn_channels,
        feat_channels=fpn_channels,
        regress_ranges=((-1, 80), (80, 160), (160, 320), (320, 1e8),),
        strides=[8, 16, 32, 64],
        center_sample_radius=1.5,
        num_joints=num_joints,
        depth_factor=1,
        z_norm=50,
        root_idx=14,
        recursive_update=dict(
            num_joints=num_joints,
            num_layers=2,
        ),
    ),
    train_cfg=dict(
        code_weight=[1.0, 1.0, 1] + [2] * num_joints * 6),
    test_cfg=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=100,
        nms_thr=0.9,
        score_thr=0.07,
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline_muco = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotationsPose3D', with_bbox=True, with_label=True),
    dict(type='ResizePose',
         scale_depth=True,
         abs_dz=abs_dz,
         img_scale=[(1280, 512), (1280, 800)], multiscale_mode='range', keep_ratio=True),
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

train_pipeline_coco = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotationsPose3D', with_bbox=True, with_label=True),
    dict(type='ResizePose',
         scale_depth=True,
         img_scale=[(1280, 512), (1280, 800)], multiscale_mode='range', keep_ratio=True),
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
        rot_range=[-0.15, 0.15],
        scale_ratio_range=[0.8, 1.2],
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
    train=[
        dict(
            type='MuCo3DHPDataset',
            use_bbox_center=use_bbox_center,
            norm_depth=True,
            abs_dz=abs_dz,
            depth_factor=1,
            data_root='data/muco/',
            ann_file='annotations/train_all_interv1.json',
            classes=('person',),
            pipeline=train_pipeline_muco,
            test_mode=False,
        ),
        dict(
            type='RepeatDataset',
            times=1,
            dataset=dict(
                type='COCOKeypointsDataset',
                convert_ids='muco',
                use_bbox_center=use_bbox_center,
                data_root='data/coco/',
                ann_file='data/coco/annotations/person_keypoints_train2017.json',
                img_prefix='data/coco/train2017/',
                classes=('person',),
                pipeline=train_pipeline_coco,
                test_mode=False,
            )
        )
    ],
    val=dict(
        type='MuPots3DHP',
        data_root='data/mupots',
        ann_file='annotations/MuPoTS-3D.json',
        norm_depth=True,
        abs_dz=abs_dz,
        depth_factor=1,
        pipeline=test_pipeline,
    ),
    test=dict(
        type='MuPots3DHP',
        data_root='data/mupots',
        ann_file='annotations/MuPoTS-3D.json',
        norm_depth=True,
        abs_dz=abs_dz,
        depth_factor=1,
        pipeline=test_pipeline,
    ),
)

optimizer = dict(
    lr=2e-3,
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type='EpochBasedRunner', max_iters=None, max_epochs=22)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=1.0 / 3,
    step=[16, 20])
log_config = dict(interval=50)
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=20,
)
evaluation = dict(interval=1)

find_unused_parameters = True

fp16 = dict(loss_scale=dict(init_scale=512))

