# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string

from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512, 512],
        help='input image size')
    args = parser.parse_args()
    return args

import math

def msa_flops(H, W, dim):
    N = H * W
    return N * dim * N * 2

def hilo_flops(H, W, l_dim, h_dim, sr_ratio):
    # H = int(N ** 0.5)
    ws = sr_ratio
    Hp = ws * math.ceil(H / ws)
    Wp = ws * math.ceil(W / ws)
    Np = Hp * Wp

    nW = (Hp // ws) * (Wp // ws)
    window_len = ws * ws
    window_flops = window_len * window_len * h_dim * 2

    high_flops = nW * window_flops
    kv_len = (Hp // sr_ratio) * (Wp // sr_ratio)
    low_flops = Np * l_dim * kv_len * 2

    return high_flops + low_flops

def get_flops(model, input_shape):
    flops, params = get_model_complexity_info(model, input_shape, as_strings=False)
    '''
    backbone = model.backbone
    backbone_name = type(backbone).__name__
    _, H, W = input_shape
    l_dim = int(backbone.alpha * backbone.num_heads[2]) * 32
    h_dim = (backbone.num_heads[2] - int(backbone.alpha * backbone.num_heads[2])) * 32
    stage3 = hilo_flops(H // 16, W // 16, l_dim, h_dim, backbone.local_ws[2]) * len(backbone.layers[2].blocks)
    stage4 = msa_flops(H // 32, W // 32, backbone.num_heads[3] * 32) * len(backbone.layers[3].blocks)
    flops += stage3 + stage4
    '''
    return flops_to_string(flops), params_to_string(params)

def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_flops(model, input_shape)
    # flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
