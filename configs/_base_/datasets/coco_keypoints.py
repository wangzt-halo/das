dataset_type = 'COCOKeypointsDataset'
data_root = 'data/coco/'
class_names = [
    'person'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsPose3D', with_bbox=True, with_label=True),
    dict(type='ResizePose', img_scale=(1600, 900), keep_ratio=True),
    dict(type='RandomFlipPose3D', flip_ratio_bev_horizontal=0.5,
         flip_pairs=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]],
         num_joints=17,),
    # dict(
    #     type='GlobalRotScaleTransPose',
    #     rot_range=[-0.78539816, 0.78539816],
    #     scale_ratio_range=[0.7, 1.3],
    #     translation_std=[0.3, 0.3],
    #     num_joints=17
    # ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundlePose3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_poses_3d',
            'gt_labels_3d', 'centers2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlipPose3D', flip_ratio_bev_horizontal=0.0,
                 flip_pairs=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]],
                 num_joints=17),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundlePose3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        use_bbox_center=True,
        data_root=data_root,
        ann_file=data_root + 'annotations/person_keypoints_train2017.json',
        img_prefix=data_root + 'train/',
        classes=class_names,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/person_keypoints_val2017.json',
        img_prefix=data_root + 'val',
        classes=class_names,
        pipeline=test_pipeline,
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/person_keypoints_val2017.json',
        img_prefix=data_root + 'val',
        classes=class_names,
        pipeline=test_pipeline,
        test_mode=False,
    ),
)
evaluation = dict(interval=2)
