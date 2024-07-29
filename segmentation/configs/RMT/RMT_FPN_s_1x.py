_base_ = [
    '../_base_/models/RMT_fpn.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.15,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, False, False],
        out_indices = (0, 1, 2, 3)
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4),
    decode_head=dict(num_classes=150))

gpu_multiplier = 1

optimizer = dict(type='AdamW', lr=0.0001*gpu_multiplier, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2
)

runner = dict(max_iters=80000, work_dir='path/RMT_FPN_s_1x')

checkpoint_config = dict(interval=4000)

evaluation = dict(interval=4000)