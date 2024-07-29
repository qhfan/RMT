_base_ = [
    '../_base_/models/FAT_fpn.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        in_chans=3,
        embed_dims=[64, 128, 256, 512],
        depths = [4, 4, 16, 4],
        kernel_sizes = [3, 5, 7, 9],
        num_heads = [2, 4, 8, 16],
        window_sizes = [8, 4, 2, 1],
        mlp_kernel_sizes = [5, 5, 5, 5],
        mlp_ratios = [4, 4, 4, 4],
        drop_path_rate = 0.12,
        use_checkpoint = False,
        out_indices = (0, 1, 2, 3)
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4),
    decode_head=dict(num_classes=150))

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)