import torch
import torch.nn as nn
from .efficientnet import Conv3DEfficientNetV2
from .transformer import TransformerEncoder
from .temporal import TCN, tcn_init
import numpy as np
import matplotlib.pyplot as plt


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


class E2E(nn.Module):
    def __init__(self, config,  dropout=0.3, in_channels=1, \
                         augmentations=False, num_classes=34, efficient_net_size="S") :
        super(E2E, self).__init__()

        self.num_classes = num_classes
        self.frontend_3d = Conv3DEfficientNetV2(config, efficient_net_size=efficient_net_size)

        self.transformer_encoder = TransformerEncoder()
        self.tcn_block = TCN()
        # tcn_init(self.tcn_block)

        self.temporal_avg = nn.AdaptiveAvgPool1d(1)

        self.fc_layer = nn.Linear(463, num_classes)
        # self.fc_layer = nn.Linear(384, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)


    # def forward(self, x, show=False, debug=False, classification=False):
    #     x = self.frontend_3d(x)
    #     if debug: print("SHAPE AFTER FRONTEND: ", x.shape)
    #     if show:
    #         plt.imshow(x[0].detach().numpy())
    #         plt.show()
    #     if debug: print("SHAPE BEFORE TRANFORMER: ", x.shape)
    #     x = self.transformer_encoder(x) # After transformer x shoud be size: Frames x 384
    #     if debug: print("SHAPE AFTER TRANSFORMER: ", x.shape)
    #     if show:
    #         plt.imshow(x[0].detach().numpy())
    #         plt.show()
    #     # x = x.unsqueeze(-1)
    #     if debug:print(x.shape)
    #     x = self.tcn_block(x) # After TCN x should be size: Frames x 463
    #     if debug:print("SHAPE AFTER TCN: ", x.shape)
        
    #     if classification:
    #         # if avg pool
    #         x = x.transpose(1, 0)
    #         x = self.temporal_avg(x)
    #         x = x.transpose(1, 0)

    #     else: x = x.transpose(2,1)
    #     x = x.squeeze()
    #     if debug: print("X SHAPE BEFOR LINEAR: ", x.shape)
    #     # x = x.permute(0, -1, 1)
    #     x = self.fc_layer(x)
    #     if debug: print("SHAPE AFTER LINEAR: ", x.shape)
    #     # if classification:
    #     #     x = self.softmax(x)
    #     #     if debug: print("SHAPE AFTER SOFTMAX: ", x.shape)
    #     # else:
    #     #     x = x.log_softmax(2)
    #     # print('SHAPE:', x.shape)

    #     return self.logsoftmax(x)

    def forward(self, x, show=False, debug=False, classification=False):
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
        if debug:print(x.shape)
        x = self.tcn_block(x) # After TCN x should be size: Frames x 463
        if debug:print("SHAPE AFTER TCN: ", x.shape)
        
        # if classification:
            # if avg pool
        x = x.transpose(1, 0)
        x = self.temporal_avg(x)
        x = x.transpose(1, 0)

        # else: x = x.transpose(2,1)
        x = x.squeeze()
        if debug: print("X SHAPE BEFOR LINEAR: ", x.shape)
        
        # x = x.permute(0, -1, 1)
        x = self.fc_layer(x)
        if debug: print("SHAPE AFTER LINEAR: ", x.shape)
        # if classification:
        #     x = self.softmax(x)
        #     if debug: print("SHAPE AFTER SOFTMAX: ", x.shape)
        # else:
        #     x = x.log_softmax(2)
        # print('SHAPE:', x.shape)
        return self.logsoftmax(x)

    





if __name__ == "__main__":
    test_tensor = torch.FloatTensor(np.random.rand(1, 29, 1, 88, 88))
    model = E2E("/home/sadevans/space/personal/LRModel/config_ef.yaml", efficient_net_size="B")
    model(test_tensor, show=False)
