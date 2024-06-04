from collections import OrderedDict
import torch
from torch import nn
from .seu import SqueezeExcite
from .conv import PointwiseConvolution, DepthwiseConv, ConvBnAct


class StochasticDepth(nn.Module):
    """StochasticDepth
    code from paper: https://link.springer.com/chapter/10.1007/978-3-319-46493-0_39

    Arguments:
        - prob: Probability of dying
        - mode: "row" or "all". "row" means that each row survives with different probability
    """
    def __init__(self, prob, mode):
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x):
        if self.prob == 0.0 or not self.training:
            return x
        else:
            shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == 'row' else [1]
            return x * torch.empty(shape).bernoulli_(self.survival).div_(self.survival).to(x.device)
        


class MBConvConfig:
    """EfficientNet Building block configuration"""
    def __init__(self, expand_ratio: float, kernel: int, stride: int, in_ch: int, out_ch: int, layers: int,
                 use_se: bool, fused: bool, act=nn.SiLU, norm_layer=nn.BatchNorm2d):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_layers = layers
        self.act = act
        self.norm_layer = norm_layer
        self.use_se = use_se
        self.fused = fused

    @staticmethod
    def adjust_channels(channel, factor, divisible=8):
        new_channel = channel * factor
        divisible_channel = max(divisible, (int(new_channel + divisible / 2) // divisible) * divisible)
        divisible_channel += divisible if divisible_channel < 0.9 * new_channel else 0
        return divisible_channel



class MBConv(nn.Module):
    """EfficientNet main building blocks

    Arguments:
        - c: MBConvConfig instance
        - sd_prob: stochastic path probability
    """
    def __init__(self, c, sd_prob=0.0):
        super(MBConv, self).__init__()
        self.inter_channel = c.adjust_channels(c.in_ch, c.expand_ratio)
        block = []
        # if c.expand_ratio == 1:
        #     block.append(('fused', ConvBnAct(c.in_ch, inter_channel, c.kernel, c.stride, 1, c.norm_layer, c.act)))
        # elif c.fused:
        if c.fused:
            ##print(c.in_ch, self.inter_channel, c.kernel, c.stride, 1, c.norm_layer, c.act)
            block.append(('fused', ConvBnAct(c.in_ch, self.inter_channel, c.kernel, c.stride, 1, c.norm_layer, c.act)))
            # block.append(('fused_point_wise', ConvBnAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity)))
            block.append(('fused_point_wise', PointwiseConvolution(self.inter_channel, c.out_ch, c.norm_layer, nn.Identity)))

        else:
            # block.append(('linear_bottleneck', ConvBnAct(c.in_ch, self.inter_channel, 1, 1, 1, c.norm_layer, c.act)))
            block.append(('linear_bottleneck', PointwiseConvolution(c.in_ch, self.inter_channel, c.norm_layer, c.act)))
            # block.append(('depth_wise', ConvBnAct(inter_channel, inter_channel, c.kernel, c.stride, inter_channel, c.norm_layer, c.act)))
            block.append(('depth_wise', DepthwiseConv(self.inter_channel, self.inter_channel, c.kernel, c.stride, self.inter_channel, c.norm_layer, c.act)))
            block.append(('se', SqueezeExcite(self.inter_channel, 4 * c.expand_ratio)))
            # block.append(('point_wise', ConvBnAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity)))
            block.append(('point_wise', PointwiseConvolution(self.inter_channel, c.out_ch, c.norm_layer, nn.Identity)))

        self.block = nn.Sequential(OrderedDict(block))
        self.use_skip_connection = c.stride == 1 and c.in_ch == c.out_ch
        self.stochastic_path = StochasticDepth(0.8, "row")

    def forward(self, x):
        # out = self.block(x)
        # out = x.clone()
        inp = x.clone()
        for bl in self.block:
            x = bl(x)
        if self.use_skip_connection:
            # out = x + self.stochastic_path(out)
            x = inp + x
        return x