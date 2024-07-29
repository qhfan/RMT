import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from typing import List
from efficientnet_pytorch.model import MemoryEfficientSwish
import torch.utils.checkpoint as checkpoint
from ..builder import BACKBONES
from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                    )
    
    def forward(self, x: torch.Tensor):
        return self.conv(x)


class PatchEmbedding(nn.Module):

    def __init__(self, in_channels=3, out_channels=96):
        super().__init__()
        self.conv1 = BasicBlock(in_channels, out_channels//2, 3, 2)
        self.conv2 = BasicBlock(out_channels//2, out_channels, 3, 2)
        self.conv3 = BasicBlock(out_channels, out_channels, 3, 1)
        self.conv4 = BasicBlock(out_channels, out_channels, 3, 1)
        self.conv5 = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        self.layernorm = nn.GroupNorm(1, out_channels)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return self.layernorm(x)

class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
                            nn.Conv2d(dim, dim, 1, 1, 0),
                            MemoryEfficientSwish(),
                            nn.Conv2d(dim, dim, 1, 1, 0)
                            #nn.Identity()
                         )
    def forward(self, x):
        return self.act_block(x)

class EfficientAttention(nn.Module):

    def __init__(self, dim, num_heads, group_split: List[int], kernel_sizes: List[int], window_size=7, 
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        #projs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2d(3*self.dim_head*group_head, 3*self.dim_head*group_head, kernel_size,
                         1, kernel_size//2, groups=3*self.dim_head*group_head))
            act_blocks.append(AttnMap(self.dim_head*group_head))
            qkvs.append(nn.Conv2d(dim, 3*group_head*self.dim_head, 1, 1, 0, bias=qkv_bias))
            #projs.append(nn.Linear(group_head*self.dim_head, group_head*self.dim_head, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1]*self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1]*self.dim_head*2, 1, 1, 0, bias=qkv_bias)
            #self.global_proj = nn.Linear(group_split[-1]*self.dim_head, group_split[-1]*self.dim_head, bias=qkv_bias)
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size!=1 else nn.Identity()

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        qkv = to_qkv(x) #(b (3 m d) h w)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous() #(3 b (m d) h w)
        q, k, v = qkv #(b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v) #(b (m d) h w)
        return res
        
    def low_fre_attention(self, x : torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        
        q = to_q(x).reshape(b, -1, self.dim_head, h*w).transpose(-1, -2).contiguous() #(b m (h w) d)
        kv = avgpool(x) #(b c h w)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, ((h//self.window_size)*(w//self.window_size))).permute(1, 0, 2, 4, 3).contiguous() #(2 b m (H W) d)
        k, v = kv #(b m (H W) d)
        attn = self.scalor * q @ k.transpose(-1, -2) #(b m (h w) (H W))
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v #(b m (h w) d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))

class PMLP(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, stride,
                 out_channels, act_layer=nn.GELU, drop_out=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.act = act_layer()
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, 
                                kernel_size//2, groups=hidden_channels)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0)
        self.drop = nn.Dropout(drop_out)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EfficientBlock(nn.Module):

    def __init__(self, dim, out_dim, num_heads, group_split: List[int], kernel_sizes: List[int], window_size: int,
                 mlp_kernel_size: int, mlp_ratio: int, stride: int, attn_drop=0., mlp_drop=0., qkv_bias=True,
                 drop_path=0.):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = EfficientAttention(dim, num_heads, group_split, kernel_sizes, window_size,
                                       attn_drop, mlp_drop, qkv_bias)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.GroupNorm(1, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.stride = stride
        if stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                                nn.Conv2d(dim, dim, mlp_kernel_size, 1, mlp_kernel_size//2),
                                nn.BatchNorm2d(dim),
                                nn.Conv2d(dim, out_dim, 1, 1, 0),
                            )
        self.mlp = PMLP(dim, mlp_hidden_dim, mlp_kernel_size, 1, out_dim, 
                        drop_out=mlp_drop)
    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.downsample(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x

class EfficientLayer(nn.Module):

    def __init__(self, depth, dim, out_dim, num_heads, group_split: List[int], kernel_sizes: List[int],
                 window_size: int, mlp_kernel_size: int, mlp_ratio: int, attn_drop=0,
                 mlp_drop=0., qkv_bias=True, drop_paths=[0., 0.], downsample=True, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.downsample = downsample
        self.blocks = nn.ModuleList(
                        [
                            EfficientBlock(dim, dim, num_heads, group_split, kernel_sizes, window_size,
                                mlp_kernel_size, mlp_ratio, 1, attn_drop, mlp_drop, qkv_bias, drop_paths[i])
                                for i in range(depth-1)
                        ]
                    )
        if downsample is True:
            self.blocks.append(EfficientBlock(dim, out_dim, num_heads, group_split, kernel_sizes, window_size,
                            mlp_kernel_size, mlp_ratio, 2, attn_drop, mlp_drop, qkv_bias, drop_paths[-1]))
        else:
            self.blocks.append(EfficientBlock(dim, out_dim, num_heads, group_split, kernel_sizes, window_size,
                            mlp_kernel_size, mlp_ratio, 1, attn_drop, mlp_drop, qkv_bias, drop_paths[-1]))

    def forward(self, x: torch.Tensor):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample:
            return F.interpolate(x, scale_factor=0.5, mode='nearest'), x
        else:
            return x, x

@BACKBONES.register_module()
class EfficientViT(nn.Module):

    def __init__(self, in_chans, embed_dims: List[int], depths: List[int],
                 num_heads: List[int], group_splits: List[List[int]], kernel_sizes: List[List[int]],
                 window_sizes: List[int], mlp_kernel_sizes: List[int], mlp_ratios: List[int],
                 attn_drop=0., mlp_drop=0., qkv_bias=True, drop_path_rate=0.1, use_checkpoint=False, out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.num_layers = len(depths)
        self.mlp_ratios = mlp_ratios
        self.patch_embed = PatchEmbedding(in_chans, embed_dims[0])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer != self.num_layers-1:
                layer = EfficientLayer(depths[i_layer], embed_dims[i_layer], embed_dims[i_layer+1], num_heads[i_layer],
                            group_splits[i_layer], kernel_sizes[i_layer], window_sizes[i_layer], 
                            mlp_kernel_sizes[i_layer], mlp_ratios[i_layer], attn_drop, mlp_drop, 
                            qkv_bias, dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], True, use_checkpoint)
            else:
                layer = EfficientLayer(depths[i_layer], embed_dims[i_layer], embed_dims[i_layer], num_heads[i_layer],
                            group_splits[i_layer], kernel_sizes[i_layer], window_sizes[i_layer], 
                            mlp_kernel_sizes[i_layer], mlp_ratios[i_layer], attn_drop, mlp_drop, 
                            qkv_bias, dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], False, use_checkpoint)
            self.layers.append(layer)
        
        for i_layer in out_indices:
            if i_layer < 3:
                layer = nn.GroupNorm(1, embed_dims[i_layer+1])
            else:
                layer = nn.GroupNorm(1, embed_dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        # print(x.size())
        x = self.patch_embed(x)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x, x_out = layer(x)
            norm_layer = getattr(self, f'norm{i}')
            x_out = norm_layer(x_out)
            outs.append(x_out)
        
        return tuple(outs)

    def train(self, mode=True):
        super(EfficientViT, self).train(mode)
