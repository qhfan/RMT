## Installation

1. Install [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

2. Download ADE20K dataset from the [official website](https://groups.csail.mit.edu/vision/datasets/ADE20K/). The directory structure should look like

   ```
   ade
   └── ADEChallengeData2016
       ├── annotations
       │   ├── training
       │   └── validation
       └── images
           ├── training
           └── validation
   ```



## Training

To train a model, run:

```bash
cd segmentation
python setup.py install
# multi-gpu training
bash tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```

For example, to train a Semantic FPN model with a FAT-B3 backbone on 8 GPUs, run:

```bash 
bash tools/dist_train.sh configs/FAT/FAT_b3.py 8 --options model.pretrained=FAT_b3.pth
```

## Benchmark

To get the FLOPs, run

```bash
python tools/get_flops.py configs/FAT/FAT_b3.py
```



## Results

#### SemanticFPN

Comparison with the state-of-the-art on
ADE20k. For B0/B1/B2, the FLOPs are measured
with input size of $512\times512$. While them for B3
are measured with $512\times2048$.

| Backbone | Params (M) | FLOPs (G) | mIoU(%) | 
|----------|------------|-----------|---------|
|FAT-B0    |8.4         |25.0       |41.5     |
|FAT-B1    |11.6        |27.5       |42.9     |
|FAT-B2    |17.2        |32.2       |45.4     |
|FAT-B3    |32.9        |179        |48.9     |

#### UperNet

| Backbone | Params (M) | FLOPs (G) | mIoU(%) | MS IoU(%) |
|----------|------------|-----------|---------|-----------|
|FAT-B3    |59          |936        |49.6     |50.7       |

