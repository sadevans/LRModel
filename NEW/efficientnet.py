import torch
import torch.nn as nn
from .efficientnet_layers.mbconv import MBConv
from .efficientnet_layers.fused_mbconv import FusedMBConv


# class MBConv(nn.Module):
#     """EfficientNet main building blocks

#     :arg
#         - c: MBConvConfig instance
#         - sd_prob: stochastic path probability
#     """
#     def init(self, c, sd_prob=0.0):
#         super(MBConv, self).init()
#         inter_channel = c.adjust_channels(c.in_ch, c.expand_ratio)
#         block = []

#         if c.expand_ratio == 1:
#             block.append(('fused', ConvBNAct(c.in_ch, inter_channel, c.kernel, c.stride, 1, c.norm_layer, c.act)))
#         elif c.fused:
#             block.append(('fused', ConvBNAct(c.in_ch, inter_channel, c.kernel, c.stride, 1, c.norm_layer, c.act)))
#             block.append(('fused_point_wise', ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity)))
#         else:
#             block.append(('linear_bottleneck', ConvBNAct(c.in_ch, inter_channel, 1, 1, 1, c.norm_layer, c.act)))
#             block.append(('depth_wise', ConvBNAct(inter_channel, inter_channel, c.kernel, c.stride, inter_channel, c.norm_layer, c.act)))
#             block.append(('se', SEUnit(inter_channel, 4 * c.expand_ratio)))
#             block.append(('point_wise', ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity)))

#         self.block = nn.Sequential(OrderedDict(block))
#         self.use_skip_connection = c.stride == 1 and c.in_ch == c.out_ch
#         self.stochastic_path = StochasticDepth(sd_prob, "row")

    # def forward(self, x):
    #     out = self.block(x)
    #     if self.use_skip_connection:
    #         out = x + self.stochastic_path(out)
    #     return out
    


class EfficientNetV2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(EfficientNetV2).__init__(*args, **kwargs)



    def _make_layer(self, channels, blocks, stride=1, se=False, fused=False):
        layers = []
        for i in range(blocks):
            if fused:
                layers.append(FusedMBConv(self.in_channels, channels, stride=stride if i == 0 else 1, se=se, fused=fused))
            else:
                layers.append(MBConv(self.in_channels, channels, stride=stride if i == 0 else 1, se=se, fused=fused))
            self.in_channels = channels
        return nn.Sequential(*layers)
