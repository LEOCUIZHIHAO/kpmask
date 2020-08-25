# optimizer
optimizer = dict(type='Adam', lr=5e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
total_epochs = 120
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[90, 110])
