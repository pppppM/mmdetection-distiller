# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import logging
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv = ConvModule(in_channels * 4, out_channels, 3,padding=1, **cfg)
        # self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))



class ResBlock(nn.Module):
    # Standard bottleneck

    def __init__(self,
                c1,
                c2,
                conv_cfg=None,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(ResBlock, self).__init__()
       
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv1 = ConvModule(c1 , c2, 1, **cfg)
        self.cv2 = ConvModule(c2 , c2, 3, padding=1,**cfg)
    

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) 



class CSPResBlock(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self,
                in_channels,
                channels,
                num_block=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(CSPResBlock, self).__init__()
        in_channels = int(in_channels)
        channels = int(channels)
        c_ = channels * 0.5
        c_ = int(c_)
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv1 = ConvModule(in_channels , c_, 1,**cfg)
        self.cv2 = nn.Conv2d(in_channels, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = ConvModule(c_ * 2 , channels, 1, **cfg)
        self.m = nn.Sequential(*[ResBlock(c_, c_) for _ in range(num_block)])
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))




class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13),conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv1 = ConvModule(c1, c_, 1,**cfg)
        self.cv2 = ConvModule(c_ * (len(k) + 1), c2, 1,**cfg)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


@BACKBONES.register_module()
class CSPDarknet(nn.Module):
    """Darknet backbone.

    Args:
        depth (int): Depth of Darknet. Currently only support 53.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.

    Example:
        >>> from mmdet.models import Darknet
        >>> import torch
        >>> self = Darknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """

    # Dict(depth: (layers, channels))
    arch_settings = {
        'depth': (3, 9, 9, 3), 
        'width':    ( (64, 128), (128, 256), (256, 512), (512, 1024)),
        'with_spp':    (False ,False , False , True),
             
        
    }

    def __init__(self,
                 depth_ratio,
                 width_ratio,
                 out_indices=(3, 4, 5),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 norm_eval=True):
        super(CSPDarknet, self).__init__()
        
        
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.depth = self.arch_settings['depth']
        self.width = self.arch_settings['width']
        self.with_spp = self.arch_settings['with_spp']
        self.depth_ratio = depth_ratio
        self.width_ratio = width_ratio
        

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.focus = Focus(3 ,int(64 * self.width_ratio) , **cfg)
       


        self.cr_blocks = ['focus']
        for i, num_blocks in enumerate(self.depth):
            layer_name = f'conv_csp_block{i + 1}'
            in_c, out_c = self.width[i]
            in_c, out_c = int(in_c * self.width_ratio), int(out_c * self.width_ratio)
            
            with_spp = self.with_spp[i]
            
            self.add_module(
                layer_name,
                self.make_conv_res_block(in_c, out_c, with_spp, int(num_blocks * self.depth_ratio),**cfg))
            self.cr_blocks.append(layer_name)
        
        self.norm_eval = norm_eval

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.cr_blocks):
            # import pdb; pdb.set_trace()
            cr_block = getattr(self, layer_name)
            x = cr_block(x)
            if i in self.out_indices:
                outs.append(x)
        

        return tuple(outs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = getattr(self, self.cr_blocks[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(CSPDarknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    @staticmethod
    def make_conv_res_block(in_channels,
                            out_channels,
                            with_spp=False,
                            num_block=1,
                            conv_cfg=None,
                            norm_cfg=dict(type='BN', requires_grad=True),
                            act_cfg=dict(type='LeakyReLU',
                                         negative_slope=0.1)):
        """In Darknet backbone, ConvLayer is usually followed by ResBlock. This
        function will make that. The Conv layers always have 3x3 filters with
        stride=2. The number of the filters in Conv layer is the same as the
        out channels of the ResBlock.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            res_repeat (int): The number of ResBlocks.
            conv_cfg (dict): Config dict for convolution layer. Default: None.
            norm_cfg (dict): Dictionary to construct and config norm layer.
                Default: dict(type='BN', requires_grad=True)
            act_cfg (dict): Config dict for activation layer.
                Default: dict(type='LeakyReLU', negative_slope=0.1).
        """

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        model = nn.Sequential()
        model.add_module(
            'conv',
            ConvModule(
                in_channels, out_channels, 3, stride=2, padding=1, **cfg))
        if with_spp:
            model.add_module('spp',SPP(out_channels,out_channels,**cfg))
            return model

        model.add_module('csp',
                            CSPResBlock(out_channels,out_channels,num_block=num_block ,**cfg))
        # for idx in range(res_repeat):
        #     model.add_module('res{}'.format(idx),
        #                      ResBlock(out_channels, **cfg))
        return model
