_base_ = [
    './kp_mask_dla34_1x_coco.py',
]
model = dict(pretrained='./model_zoo/ctdet_coco_dla_2x_mmdet.pth',
                backbone=dict(heads=dict(hm=80, wh=4, reg=2)))
train_cfg = dict(
    dynamic_training=True,
    gt_debug=False)
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3
)
optimizer = dict(type='Adam', lr=5e-4)
