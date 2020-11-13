# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import NECKS


class Block(nn.Module):
    # Standard bottleneck

    def __init__(self,
                c1,
                c2,
                conv_cfg=None,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(Block, self).__init__()
        
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv1 = ConvModule(c1 , c2, 1, **cfg)
        self.cv2 = ConvModule(c2 , c2, 3, padding=1,**cfg)
    

    def forward(self, x):
        return  self.cv2(self.cv1(x)) 

class CSPBlock(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self,
                in_channels,
                channels,
                expansion=0.5,
                num_block=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(CSPBlock, self).__init__()
        in_channels = int(in_channels)
        channels = int(channels)
        c_ = channels * expansion
        c_ = int(c_)
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv1 = ConvModule(in_channels , c_, 1, **cfg)
        self.cv2 = nn.Conv2d(in_channels, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = ConvModule(c_ * 2 , channels, 1, **cfg)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Block(c_, c_) for _ in range(num_block)])


    def forward(self, x):
        # import pdb; pdb.set_trace()
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))



@NECKS.register_module()
class YOLOV5Neck(nn.Module):
    """The neck of YOLOV3.

    It can be treated as a simplified version of FPN. It
    will take the result from Darknet backbone and do some upsampling and
    concatenation. It will finally output the detection result.

    Note:
        The input feats should be from top to bottom.
            i.e., from high-lvl to low-lvl
        But YOLOV3Neck will process them in reversed order.
            i.e., from bottom (high-lvl) to top (low-lvl)

    Args:
        num_scales (int): The number of scales / stages.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """
    arch_settings = dict(
        bottom_top_channels= ((1024,1024,3),(1024,512,3)),
        top_bottom_channels= ((512,256,3),(512,512,3),(1024,1024,3)),
    )



    def __init__(self,
                 num_scales,
                 depth_ratio,
                 width_ratio,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(YOLOV5Neck, self).__init__()
        # assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = num_scales
        
        self.depth_ratio = depth_ratio
        self.width_ratio = width_ratio
        

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # To support arbitrary scales, the code looks awful, but it works.
        # Better solution is welcomed.
        
        
        for i,(in_c,out_c ,depth) in enumerate( self.arch_settings['bottom_top_channels']):
            in_c, out_c = int(in_c * self.width_ratio), int(out_c * self.width_ratio)
            self.add_module(f'conv_bottom_top{i}', ConvModule(out_c, out_c//2, 1, **cfg))
            # in_c + out_c : High-lvl feats will be cat with low-lvl feats
            self.add_module(f'csp_bottom_top{i}',
                            CSPBlock(in_c , out_c,
                            num_block=int(depth * depth_ratio), **cfg))
        for i,(in_c,out_c ,depth)  in  enumerate(self.arch_settings['top_bottom_channels']):

            in_c, out_c = int(in_c * self.width_ratio), int(out_c * self.width_ratio)
            if i < self.num_scales - 1:
                self.add_module(f'conv_top_bottom{i}', ConvModule(out_c , out_c, 3,stride=2,padding=1, **cfg))
            # in_c + out_c : High-lvl feats will be cat with low-lvl feats
            self.add_module(f'csp_top_bottom{i}',
                            CSPBlock(in_c , out_c,
                            num_block=int(depth * depth_ratio),**cfg))
        


    def forward(self, feats):
        assert len(feats) == self.num_scales
        # import pdb;pdb.set_trace()
        # processed from bottom (high-lvl) to top (low-lvl)
        bottom_top = []
        bottom_feat = feats[-1]

        for i, x in enumerate(reversed(feats[:-1])):
            # import pdb; pdb.set_trace()
            conv = getattr(self, f'conv_bottom_top{i}')
            csp = getattr(self, f'csp_bottom_top{i}')


            bottom_feat = conv(csp(bottom_feat))
            bottom_top.append(bottom_feat)

            bottom_feat = F.interpolate(bottom_feat, scale_factor=2)
            bottom_feat = torch.cat((bottom_feat, x), 1)

        top_bottom = []
        top_feat = bottom_feat
        # import pdb; pdb.set_trace()
        for i, x in enumerate(bottom_top[::-1]):
            conv = getattr(self, f'conv_top_bottom{i}')
            csp = getattr(self, f'csp_top_bottom{i}')
            # import pdb; pdb.set_trace()

            top_feat = csp(top_feat)
            top_bottom.append(top_feat)

            top_feat = conv(top_feat)
            top_feat = torch.cat((top_feat, x), 1)

        last_csp = getattr(self, f'csp_top_bottom{i+1}')
        top_bottom.append(last_csp(top_feat))


        return tuple(top_bottom[::-1])

    def init_weights(self):
        """Initialize the weights of module."""
        # init is done in ConvModule
        pass
