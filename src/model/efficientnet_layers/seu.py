from torch import nn
from functools import partial


class SqueezeExcite(nn.Module):
    """Squeeze-Excitation Unit

    paper: https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper

    """
    def __init__(self, in_channel, reduction_ratio=24, act1=partial(nn.SiLU, inplace=True), act2=nn.Sigmoid):
        super(SqueezeExcite, self).__init__()
        reduced_channels = in_channel // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channel, reduced_channels, (1, 1), bias=True)
        self.fc2 = nn.Conv2d(reduced_channels, in_channel, (1, 1), bias=True)
        self.act1 = act1()
        self.act2 = act2()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.act1(y)
        y = self.fc2(y)
        y = self.act2(y)
        return x * y