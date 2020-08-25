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
        type='KeyPointMaskRoIHead',
        num_stages=1,
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=14, sample_num=0),
            out_channels=256,
            featmap_strides=[4]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(
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
    mask_size=28,
    gt_debug=False)
test_cfg = dict(
    K=100,
    stack_out_level=[0], #[0]->use 2st stack, [1]->use 1st stack, [0, 1]-> merge 2 stacks
    score_thr=0.05,
    mask_thr_binary=0.5,
    nms=dict(type='nms', iou_thr=0.9),
    )
