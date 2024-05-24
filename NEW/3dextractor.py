import torch
import torch.nn as nn

def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


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
    def __init__(self, in_channels, out_channels, kernel=(3, 5, 5), loss_type='relu', if_maxpool=False) -> None:
        super(Conv3D).__init__()
        self.if_maxpool = if_maxpool
        if loss_type == "relu":
            self.act = nn.ReLU()
        elif loss_type == 'swish':
            self.act = Swish()

        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel, (1, 2, 2), (1, 2, 2)),
        self.bn = nn.BatchNorm3d(self.frontend_nout),

        if self.if_maxpool:
            self.maxpool = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))


    def forward(self, x):
        x = x.transpose(1, 2)  # [B, T, C, H, W] -> [B, C, T, H, W]

        B, C, T, H, W = x.size()
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.act(x)

        if self.if_maxpool:
            x = self.maxpool(x)

        return x
    

# if __name__ == "__main__":

