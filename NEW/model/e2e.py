import torch
import torch.nn as nn
from efficientnet import EfficientNetV2, get_efficientnet_v2
from frontend import Conv3D, get_conv_3d
from transformer import TransformerEncoder
from temporal import TCN
import numpy as np
import matplotlib.pyplot as plt


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


class E2E(nn.Module):
    def __init__(self, config,  efficient_net_size="S") :
        super(E2E, self).__init__()

        num_classes = 500
        # self.frontend_3d = Conv3D()
        self.frontend_3d = get_conv_3d(config, model_size=efficient_net_size)

        # self.frontend = EfficientNetV2(args.layers_info)
        self.frontend = get_efficientnet_v2(config, model_size=efficient_net_size)
        self.transformer_encoder = TransformerEncoder()
        self.tcn_block = TCN()


        self.temporal_avg = nn.AdaptiveAvgPool1d(1)

        # self.fc_layer = nn.Linear(463, num_classes)


        # self.criterion = 
    def forward(self, x, show=False):
        print("INPUT SHAPE: ", x.shape)
        x = self.frontend_3d(x) # After frontend x shoud be size: Frames x 24 x 44 x 44
        print("SHAPE AFTER FRONTEND 3D: ", x.shape)
        Tnew = x.shape[2]
        x = threeD_to_2D_tensor(x)
        print("SHAPE AFTER CONVERTION INTO 2D: ", x.shape)
        x = self.frontend(x) # After frontend x shoud be size: Frames x 384
        print("SHAPE AFTER FRONTEND: ", x.shape)
        x = x.squeeze(2,3)
        # x = x.squeeze(2)

        print(x)

        if show:
            plt.imshow(x.detach().numpy())
            plt.show()
        print("SHAPE BEFORE TRANFORMER: ", x.shape)
        x = self.transformer_encoder(x) # After transformer x shoud be size: Frames x 384
        print("SHAPE AFTER TRANSFORMER: ", x.shape)
        if show:
            plt.imshow(x.detach().numpy())
            plt.show()
        x = x.unsqueeze(-1)
        print(x.shape)
        x = self.tcn_block(x) # After TCN x should be size: Frames x 463
        print("SHAPE AFTER TCN: ", x.shape)
        x = x.transpose(1, 0)
        x = self.temporal_avg(x)
        x = x.transpose(1, 0)

        print(x.shape)
        return x


if __name__ == "__main__":
    test_tensor = torch.FloatTensor(np.random.rand(1, 29, 1, 88, 88))
    # test_tensor = torch.FloatTensor(np.random.rand(29, 1, 88, 88))

    print(test_tensor.shape)
    model = E2E("/home/sadevans/space/personal/LRModel/config_ef.yaml", efficient_net_size="B")
    model(test_tensor, show=False)
