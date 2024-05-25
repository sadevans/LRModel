import torch
import torch.nn as nn
import yaml
import numpy as np




# class Conv3dResNet(torch.nn.Module):
#     """Conv3dResNet module"""

#     def __init__(self, backbone_type="resnet", relu_type="swish"):
#         """__init__.

#         :param backbone_type: str, the type of a visual front-end.
#         :param relu_type: str, activation function used in an audio front-end.
#         """
#         super(Conv3dResNet, self).__init__()
#         # self.run_mode = run_mode
#         self.frontend_nout = 64
#         self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
#         self.frontend3D = nn.Sequential(
#             nn.Conv3d(
#                 1, self.frontend_nout, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False
#             ),
#             nn.BatchNorm3d(self.frontend_nout),
#             Swish(),
#             nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
#         )

#     def forward(self, xs_pad):
#         xs_pad = xs_pad.transpose(1, 2)  # [B, T, C, H, W] -> [B, C, T, H, W]

#         B, C, T, H, W = xs_pad.size()
#         xs_pad = self.frontend3D(xs_pad)
#         Tnew = xs_pad.shape[2]
#         xs_pad = threeD_to_2D_tensor(xs_pad)
#         xs_pad = self.trunk(xs_pad)
#         return xs_pad.view(B, Tnew, xs_pad.size(1))


class Swish(nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)  
    
 

class Conv3D(nn.Module):
    """Convolution 3D block"""
    def __init__(self, in_channels=1, out_channels=24, kernel=(3, 5, 5), loss_type='relu', if_maxpool=False):
        super(Conv3D, self).__init__()
        #print('here: ', in_channels, out_channels, kernel, loss_type, if_maxpool)
        self.if_maxpool = if_maxpool
        if loss_type == "relu":
            self.act = nn.ReLU()
        elif loss_type == 'swish':
            self.act = Swish()

        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel, (1, 2, 2), (1, 2, 2))
        self.bn = nn.BatchNorm3d(out_channels)

        if self.if_maxpool:
            self.maxpool = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))


    def forward(self, x):
        #print("INPUT SHAPE IN 3D CONV: ", x.shape)
        x = x.transpose(1, 2)  # [B, T, C, H, W] -> [B, C, T, H, W]
        #print("INPUT SHAPE IN 3D CONV AFTER TRANSPOSE: ", x.shape)

        # B, C, T, H, W = x.size()
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.act(x)

        if self.if_maxpool:
            x = self.maxpool(x)

        #print("SHAPE AFTER 3D CONV: ", x.shape)
        # x = x.transpose(2, 1)
        # #print("SHAPE AFTER 3D CONV TRANSPOSE: ", x.shape)

        return x


def get_conv_3d(config, model_size="S"):
    with open(config, 'r') as file:
        info = yaml.safe_load(file)
    info_el = info['frontend-3d'][0]
    out_channels = info['efficient-net-blocks'][model_size][0][3]

    return Conv3D(in_channels=info_el[0], out_channels=out_channels, kernel=tuple(info_el[2]), loss_type=info_el[3], if_maxpool=info_el[4])



if __name__ == "__main__":
    with open('/home/sadevans/space/personal/LRModel/config_ef.yaml', 'r') as file:
        info = yaml.safe_load(file)
    # #print(info['frontend-3d'])

    test_tensor = torch.FloatTensor(np.random.rand(4, 29, 1, 88, 88))
    #print(test_tensor.shape)
    for info_el in info['frontend-3d']:
        #print(info_el)
        c = Conv3D(in_channels=info_el[0], out_channels=info_el[1], kernel=tuple(info_el[2]), loss_type=info_el[3], if_maxpool=info_el[4])
        out = c(test_tensor)


    # kernel = 

