import torch.nn as nn
from collections import OrderedDict
from .efficientnet_layers.mbconv import MBConv, MBConvConfig
from .frontend import get_conv_3d
import copy
import yaml

    
class EfficientNetV2(nn.Module):
    """Pytorch Implementation of EfficientNetV2

    paper: https://arxiv.org/abs/2104.00298

    - reference 1 (pytorch): https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
    - reference 2 (official): https://github.com/google/automl/blob/master/efficientnetv2/effnetv2_configs.py

    :arg
        - layer_infos: list of MBConvConfig
        - out_channels: bottleneck channel
        - nlcass: number of class
        - dropout: dropout probability before classifier layer
        - stochastic depth: stochastic depth probability
    """
    def __init__(self, layer_infos, out_channels=384, dropout=0.3, stochastic_depth=0.8,
                 block=MBConv, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super(EfficientNetV2, self).__init__()
        self.layer_infos = layer_infos
        self.norm_layer = norm_layer
        self.act = act_layer

        self.in_channel = layer_infos[0].in_ch
        self.final_stage_channel = layer_infos[-1].out_ch

        self.cur_block = 0
        self.num_block = sum(stage.num_layers for stage in layer_infos)
        self.stochastic_depth = stochastic_depth

        self.blocks = nn.Sequential(*self.make_stages(layer_infos, block))

       
        self.stage_7 = nn.Sequential(OrderedDict([
            ('pointwise', nn.Conv2d(self.final_stage_channel, 768, kernel_size=1, stride=1, padding=0, groups=1, bias=False)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('GLU', nn.GLU(dim=1))
        ]))

    def make_stages(self, layer_infos, block):
        return [layer for layer_info in layer_infos for layer in self.make_layers(copy.copy(layer_info), block)]

    def make_layers(self, layer_info, block):
        layers = []
        for i in range(layer_info.num_layers):
            layers.append(block(layer_info, sd_prob=self.get_sd_prob()))
            layer_info.in_ch = layer_info.out_ch
            layer_info.stride = 1
        return layers

    def get_sd_prob(self):
        sd_prob = self.stochastic_depth * (self.cur_block / self.num_block)
        self.cur_block += 1
        return sd_prob

    def forward(self, x, debug=False):
        """forward.

        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """
        if debug: print("INPUT SHAPE IN EFFICIENT NET V2: ", x.shape)
        for i, block in enumerate(self.blocks):
            if debug: print(f"NOW IN BLOCK {i}: ", block)
            x = block(x)
            if debug: print(f"SHAPE AFTER BLOCK {i}: ", x.shape)

        x = self.stage_7(x)

        return x

def efficientnet_v2_init(model):
    for m in model.modules():
        # if isinstance(m, nn.Conv2d):
        #     nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #     if m.bias is not None:
        #         nn.init.zeros_(m.bias)
        if isinstance(m, (nn.BatchNorm2d)):
            # nn.init.ones_(m.weight)
            # nn.init.zeros_(m.bias)
            m.momentum = 0.99
        # elif isinstance(m, nn.Linear):
        #     nn.init.normal_(m.weight, mean=0.0, std=0.01)
        #     nn.init.zeros_(m.bias)


def get_efficientnet_v2(config, model_size="B", pretrained=False, dropout=0.3, stochastic_depth=0.8, **kwargs):
    residual_config = [MBConvConfig(*layer_config) for layer_config in get_efficientnet_v2_structure(config, model_size)]
    model = EfficientNetV2(residual_config, dropout=dropout, stochastic_depth=stochastic_depth, block=MBConv, act_layer=nn.SiLU)
    efficientnet_v2_init(model)

    return model


def get_efficientnet_v2_structure(config, model_size='B'):
    with open(config, 'r') as file:
        info = yaml.safe_load(file)
    efficientnet_config = info['efficient-net-blocks'][model_size]
    return info['efficient-net-blocks'][model_size]


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)



class Conv3DEfficientNetV2(nn.Module):
    def __init__(self, config, efficient_net_size="B"):
        super(Conv3DEfficientNetV2, self).__init__()
        self.conv3d = get_conv_3d(config, model_size=efficient_net_size)
        self.efnet = get_efficientnet_v2(config, model_size=efficient_net_size)

    def forward(self, x, show=False, debug=False):
        # print('SHAPE: ', x.shape)
        # if (x.shape[1] != 1 or x.shape[1] != 3) and (x.shape[2] == 1 or x.shape[2] == 3):
        #     x = x.permute(0, 2, 1, 3, 4)

        B, C, T, H, W = x.shape
        if debug: print("INPUT SHAPE: ", x.shape)
        x = self.conv3d(x)                                         # After efnet x shoud be size: Frames x Channels x H x W
        if debug: print("SHAPE AFTER FRONTEND 3D: ", x.shape)
        Tnew = x.shape[2]
        x = threeD_to_2D_tensor(x)
        
        if debug: print("SHAPE AFTER CONVERTION INTO 2D: ", x.shape)
        x = self.efnet(x)                                            # After efnet x shoud be size: Frames x 384
        if debug: print("SHAPE AFTER FRONTEND: ", x.shape)
        x = x.view(B, Tnew, x.size(1))

        return x




if __name__ == "__main__":
    model = get_efficientnet_v2()
    efficientnet_v2_init(model)