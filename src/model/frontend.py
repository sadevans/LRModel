import torch
import torch.nn as nn
import yaml
import numpy as np


class Swish(nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)  
    
 

class Conv3D(nn.Module):
    """Convolution 3D block"""
    def __init__(self, in_channels=1, out_channels=24, kernel=(3, 5, 5), loss_type='relu', if_maxpool=False):
        super(Conv3D, self).__init__()
        ##print('here: ', in_channels, out_channels, kernel, loss_type, if_maxpool)
        self.if_maxpool = if_maxpool
        if loss_type == "relu":
            self.act = nn.ReLU()
        elif loss_type == 'swish':
            self.act = Swish()

        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel, (1, 2, 2), (1, 2, 2))
        self.bn = nn.BatchNorm3d(out_channels, momentum=0.99)

        if self.if_maxpool:
            self.maxpool = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))


    def forward(self, x, debug=True):
        # if debug: print("INPUT SHAPE IN 3D CONV: ", x.shape)
        # if (x.shape[1] != 1 or x.shape[1] != 3) and (x.shape[2] == 1 or x.shape[2] == 3):
        #     x = x.permute(0, 2, 1, 3, 4)
        # else:
        #     x = x.transpose(1, 2)  # [B, T, C, H, W] -> [B, C, T, H, W]
        #if debug: #print("INPUT SHAPE IN 3D CONV AFTER TRANSPOSE: ", x.shape)

        # B, C, T, H, W = x.size()
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.act(x)

        if self.if_maxpool:
            x = self.maxpool(x)

        #if debug: #print("SHAPE AFTER 3D CONV: ", x.shape)
        return x


def init_3dconv(model):
    for m in model.modules():
        # if isinstance(m, nn.Conv2d):
        #     nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #     if m.bias is not None:
        #         nn.init.zeros_(m.bias)
        if isinstance(m, (nn.BatchNorm3d)):
            # nn.init.ones_(m.weight)
            # nn.init.zeros_(m.bias)
            m.momentum = 0.99
        # elif isinstance(m, nn.Linear):
        #     nn.init.normal_(m.weight, mean=0.0, std=0.01)
        #     nn.init.zeros_(m.bias)


def get_conv_3d(config, model_size="S"):
    with open(config, 'r') as file:
        info = yaml.safe_load(file)
    info_el = info['frontend-3d'][0]
    out_channels = info['efficient-net-blocks'][model_size][0][3]
    model = Conv3D(in_channels=info_el[0], out_channels=out_channels, kernel=tuple(info_el[2]), loss_type=info_el[3], if_maxpool=info_el[4])
    init_3dconv(model)
    return model



if __name__ == "__main__":
    with open('/home/sadevans/space/personal/LRModel/config_ef.yaml', 'r') as file:
        info = yaml.safe_load(file)
    # ##print(info['frontend-3d'])

    test_tensor = torch.FloatTensor(np.random.rand(4, 29, 1, 88, 88))
    ##print(test_tensor.shape)
    for info_el in info['frontend-3d']:
        ##print(info_el)
        c = Conv3D(in_channels=info_el[0], out_channels=info_el[1], kernel=tuple(info_el[2]), loss_type=info_el[3], if_maxpool=info_el[4])
        out = c(test_tensor)


    # kernel = 

