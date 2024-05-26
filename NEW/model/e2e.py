import torch
import torch.nn as nn
from .efficientnet import Conv3DEfficientNetV2
from .transformer import TransformerEncoder
from .temporal import TCN
import numpy as np
import matplotlib.pyplot as plt


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


class E2E(nn.Module):
    def __init__(self, config,  num_classes=34, efficient_net_size="S") :
        super(E2E, self).__init__()

        self.num_classes = num_classes
        self.frontend_3d = Conv3DEfficientNetV2(config, efficient_net_size=efficient_net_size)

        self.transformer_encoder = TransformerEncoder()
        self.tcn_block = TCN()


        self.temporal_avg = nn.AdaptiveAvgPool1d(1)

        # self.fc_layer = nn.Linear(463, num_classes)
        self.fc_layer = nn.Linear(384, num_classes)

        self.softmax = nn.Softmax(dim=1)

        # self.fc_layer = nn.Linear(463, num_classes)


    def forward(self, x, show=False, debug=False):
        x = self.frontend_3d(x)
        if debug: print("SHAPE AFTER FRONTEND: ", x.shape)
        if show:
            plt.imshow(x[0].detach().numpy())
            plt.show()
        if debug: print("SHAPE BEFORE TRANFORMER: ", x.shape)
        x = self.transformer_encoder(x) # After transformer x shoud be size: Frames x 384
        if debug: print("SHAPE AFTER TRANSFORMER: ", x.shape)
        if show:
            plt.imshow(x[0].detach().numpy())
            plt.show()
        # x = x.unsqueeze(-1)
        # #print(x.shape)
        # x = self.tcn_block(x) # After TCN x should be size: Frames x 463
        if debug: print("SHAPE AFTER TCN: ", x.shape)
        # x = x.transpose(1, 0)
        # x = x.transpose(2,1)
        # x = self.temporal_avg(x)
        # x = x.transpose(1, 0).squeeze()
        if debug: print("X SHAPE BEFOR LINEAR: ", x.shape)
        x = self.fc_layer(x)
        if debug: print("SHAPE AFTER LINEAR: ", x.shape)
        # x = self.softmax(x)
        #if debug: print("SHAPE AFTER SOFTMAX: ", x.shape)

        ##print(x.shape)
        return x


if __name__ == "__main__":
    test_tensor = torch.FloatTensor(np.random.rand(1, 29, 1, 88, 88))
    model = E2E("/home/sadevans/space/personal/LRModel/config_ef.yaml", efficient_net_size="B")
    model(test_tensor, show=False)
