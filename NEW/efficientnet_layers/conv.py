import torch.nn as nn


class PointwiseConvolution(nn.Module):
    """Pointwise-Convolution-BatchNormalization-Activation Module"""
    def __init__(self, in_channels, out_channels, norm_layer, act):
        super(PointwiseConvolution, self).__init__(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
        norm_layer(out_channels),
        act()
        )

class DepthwiseConv(nn.Module):
    """Depthwise-Convolution-BatchNormalization-Activation Module"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, norm_layer, act):
        super(DepthwiseConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=groups, bias=False),
            norm_layer(out_channels),
            act()
        )


class ConvBnAct(nn.Module):
    """Convolution-BatchNormalization-Activation Module"""
    def __init__(self, in_channel, out_channel, kernel_size, stride, groups, norm_layer, act, conv_layer=nn.Conv2d):
        super(ConvBnAct, self).__init__(
            conv_layer(in_channel, out_channel, kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=groups, bias=False),
            norm_layer(out_channel),
            act()
        )