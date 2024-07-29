_base_ = [
    '../_base_/models/FAT_fpn.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        in_chans=3,
        embed_dims=[48, 96, 192, 384],
        depths = [2, 2, 6, 2],
        kernel_sizes = [3, 5, 7, 9],
        num_heads = [3, 6, 12, 24],
        window_sizes = [8, 4, 2, 1],
        mlp_kernel_sizes = [5, 5, 5, 5],
        mlp_ratios = [4, 4, 4, 4],
        drop_path_rate = 0.0,
        use_checkpoint = False,
        out_indices = (0, 1, 2, 3)
    ),
    neck=dict(
        type='FPN',
        in_channels=[48, 96, 192, 384],
        out_channels=256,
        num_outs=4),
    decode_head=dict(num_classes=150))

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)