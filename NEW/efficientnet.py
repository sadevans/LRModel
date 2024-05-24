import torch
import torch.nn as nn
from .efficientnet_layers.mbconv import MBConv, MBConvConfig
from frontend import Conv3D
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
    def __init__(self, layer_infos, out_channels=1280, nclass=0, dropout=0.2, stochastic_depth=0.0,
                 block=MBConv, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super(EfficientNetV2, self).__init__()
        self.layer_infos = layer_infos
        self.norm_layer = norm_layer
        self.act = act_layer

        self.in_channel = layer_infos[0].in_ch
        self.final_stage_channel = layer_infos[-1].out_ch
        self.out_channels = out_channels

        self.cur_block = 0
        self.num_block = sum(stage.num_layers for stage in layer_infos)
        self.stochastic_depth = stochastic_depth

        self.frontend3d = Conv3D(1, 24)
        self.blocks = nn.Sequential(*self.make_stages(layer_infos, block))

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

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

    def change_dropout_rate(self, p):
        self.head[-2] = nn.Dropout(p=p, inplace=True)


def efficientnet_v2_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.zeros_(m.bias)


def get_efficientnet_v2(model_name, pretrained, nclass=0, dropout=0.1, stochastic_depth=0.2, **kwargs):
    residual_config = [MBConvConfig(*layer_config) for layer_config in get_efficientnet_v2_structure(model_name)]
    model = EfficientNetV2(residual_config, 1280, nclass, dropout=dropout, stochastic_depth=stochastic_depth, block=MBConv, act_layer=nn.SiLU)
    efficientnet_v2_init(model)

    # if pretrained:
    #     load_from_zoo(model, model_name)

    return model





def get_efficientnet_v2_structure(model_name):
    with open(f'{model_name}.yaml', 'r') as params_data:
        data_loaded = yaml.safe_load(params_data)



class Conv3dEfficientNetV2(nn.Module):
    def __init__(self, layer_infos, out_channels=384, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d, dropout=0.2, stochastic_depth=0.0,):
        super(Conv3dEfficientNetV2).__init__()
        self.layer_infos = layer_infos
        self.norm_layer = norm_layer
        self.act = act_layer

        self.out_channels = out_channels
        self.cur_block = 0
        self.num_block = sum(stage.num_layers for stage in layer_infos)
        self.stochastic_depth = stochastic_depth

        self.frontend3D = Conv3D(in_channels=1, out_channels=24)
        self.blocks = nn.Sequential(*self.make_stages(layer_infos))


    def make_stages(self, layer_infos, block):
        return [layer for layer_info in layer_infos for layer in self.make_layers(copy.copy(layer_info), block)]
    

    def make_layers(self, layer_info, block):
        layers = []
        for i in range(layer_info.num_layers):
            layers.append(block(layer_info, sd_prob=self.get_sd_prob()))
            layer_info.in_ch = layer_info.out_ch
            layer_info.stride = 1
        return layers
