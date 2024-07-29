_base_ = [
    '../_base_/models/RMT_fpn.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[80, 160, 320, 512],
        depths=[4, 8, 25, 8],
        num_heads=[5, 5, 10, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[5, 5, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.4,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        out_indices = (0, 1, 2, 3)
    ),
    neck=dict(
        type='FPN',
        in_channels=[80, 160, 320, 512],
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

runner = dict(max_iters=80000, work_dir='path/RMT_FPN_m_1x')

checkpoint_config = dict(interval=4000)

evaluation = dict(interval=4000)