dataset_type = 'CocoDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='ShiftScaleRotate',
        shift_range=[-30, 30],
        scale_range=[0.75, 1.3],
        rotate_range=[-30, 30],
        remain_ratio=0.3,
        aug_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2014.json',
        img_prefix=data_root + 'train2014/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2014.json',
        img_prefix=data_root + 'val2014/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2014.json',
        img_prefix=data_root + 'val2014/',
        pipeline=test_pipeline))

evaluation = dict(interval=5, metric=['bbox', 'segm'])
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
total_epochs = 130
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    warmup_by_epoch=True,
    step=[90, 110])
checkpoint_config = dict(interval=5, create_symlink=False)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='Keypoint_mask',
    pretrained='../model_zoo/ctdet_coco_dla_2x_mmdet.pth',
    backbone=dict(type='DLASeg', heads=dict(hm=1, wh=2, reg=2)),
    bbox_head=dict(
        type='KeyPointHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        stacked_convs=1,
        deconv_method=None,
        loss_cls=dict(
            type='KeyPointFocalLoss', alpha=2, beta=4, loss_weight=1.0),
        loss_bbox=dict(type='KeyPointRegL1Loss', loss_weight=0.1),
        loss_offset=dict(type='KeyPointRegL1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='KeyPointRendRoIHead',
        mask_roi_extractor=dict(
            type='GenericRoIExtractor',
            aggregation='concat',
            roi_layer=dict(type='SimpleRoIAlign', out_size=14),
            out_channels=256,
            featmap_strides=[4]),
        mask_head=dict(
            type='CoarseMaskHead',
            num_fcs=2,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        point_head=dict(
            type='KeyPointMaskHead',
            num_fcs=3,
            in_channels=256,
            fc_channels=256,
            num_classes=1,
            coarse_pred_each_layer=True,
            loss_point=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))
train_cfg = dict(
    num_classes=1,
    max_objs=100,
    dynamic_training=None,
    opt=dict(dense_wh=False, cat_spec_wh=False),
    gt_debug=False,
    mask_size=7,
    num_points=196,
    oversample_ratio=3,
    importance_sample_ratio=0.75)
test_cfg = dict(
    K=100,
    stack_out_level=[0],
    score_thr=0.05,
    mask_thr_binary=0.5,
    nms=dict(type='nms', iou_thr=0.9),
    subdivision_steps=5,
    subdivision_num_points=784,
    scale_factor=2)
work_dir = 'projects/'
