_base_ = [
    '../_base_/models/RMT_retinanet.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        embed_dims=[112, 224, 448, 640],
        depths=[4, 8, 25, 8],
        num_heads=[7, 7, 14, 20],
        init_values=[2, 2, 2, 2],
        heads_ranges=[6, 6, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.5,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        out_indices = (0, 1, 2, 3)
    ),
    neck=dict(
        type='FPN',
        in_channels=[112, 224, 448, 640],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5)
)


optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

runner = dict(type='EpochBasedRunner', max_epochs=12)
fp16 = dict(loss_scale=512.0)
find_unused_parameters=True
###########################################################################################################

# place holder for new verison mmdet compatiability
resume_from=None

# custom
checkpoint_config = dict(max_keep_ckpts=1)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)