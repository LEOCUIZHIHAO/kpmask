_base_ = [
    '../_base_/datasets/coco2017_instance.py',
    '../_base_/schedules/schedule_120e.py', '../_base_/default_runtime.py'
]
# model settings
num_classes=80
model = dict(
    type='Keypoint_mask',
    pretrained='./model_zoo/ctdet_coco_dla_2x_mmdet80.pth',
    backbone=dict(
        type='DLASeg',
        heads=dict(hm=num_classes, wh=2, reg=2)),
    bbox_head=dict(
        type='KeyPointHead',
        num_classes=num_classes,
        in_channels=256,
        feat_channels=256,
        stacked_convs=1,
        deconv_method=None,
        loss_cls=dict(type='KeyPointFocalLoss', alpha=2, beta=4, loss_weight=1.0),
        loss_bbox=dict(type='KeyPointRegL1Loss', loss_weight=0.1),
        loss_offset=dict(type='KeyPointRegL1Loss', loss_weight=1.0),
        ),
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
            num_classes=num_classes,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        point_head=dict(
            type='KeyPointMaskHead',
            num_fcs=3,
            in_channels=256,
            fc_channels=256,
            num_classes=num_classes,
            coarse_pred_each_layer=True,
            loss_point=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))
# model training and testing settings
train_cfg = dict(
    num_classes=num_classes,
    max_objs=100,
    dynamic_training=None,
    opt=dict(
             dense_wh=False,
             cat_spec_wh=False,
             ),
    mask_size=7,
    num_points=14 * 14,
    oversample_ratio=3,
    importance_sample_ratio=0.75,
    gt_debug=False,
    )
test_cfg = dict(
    K=100,
    stack_out_level=[0], #[0]->use 2st stack, [1]->use 1st stack, [0, 1]-> merge 2 stacks
    score_thr=0.05,
    mask_thr_binary=0.5,
    nms=dict(type='nms', iou_thr=0.9),
    subdivision_steps=5,
    subdivision_num_points=28 * 28,
    scale_factor=2)
