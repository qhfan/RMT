# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='FAT',
        in_chans=3,
        embed_dims=[32, 80, 160, 256],
        depths = [2, 2, 6, 2],
        kernel_sizes = [3, 5, 7, 9],
        num_heads = [2, 5, 10, 16],
        window_sizes = [8, 4, 2, 1],
        mlp_kernel_sizes = [5, 5, 5, 5],
        mlp_ratios = [4, 4, 4, 4],
        drop_path_rate = 0.05,
        use_checkpoint = False,
        out_indices = (0, 1, 2, 3)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
