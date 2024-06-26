import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np


class MyModel(torch.nn.Module):
    def __init__(self, dropout_p=0.5):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))     
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.gru1  = nn.GRU(96*5*5, 256, 1, bidirectional=True)
        self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)
        
        self.FC    = nn.Linear(512, 33+1)
        self.dropout_p  = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)        
        self.dropout3d = nn.Dropout3d(self.dropout_p)  
        # self._init()

    
    def _init(self):
        
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)
        
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.constant_(self.conv2.bias, 0)
        
        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        init.constant_(self.conv3.bias, 0)        
        
        init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
        init.constant_(self.FC.bias, 0)
        
        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(m.weight_ih_l0[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + 256])
                init.constant_(m.bias_ih_l0[i: i + 256], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)
        
        
    def forward(self, x):
        # ##print('IN THE INPUT MODEL: ', x)
        # ##print('INPUT IN MODEL SHAPE: ', x.shape)
        x = self.conv1(x)
        # ##print('AFTER FIST CONV: ', x)
        ##print('CONV1 WEIGHTS: ', self.conv1.weight)
        # ##print('SHAPE AFTER FIRST 3D CONV: ', x.shape)
        x = self.relu(x)
        # ##print(x)
        # ##print('SHAPE AFTER RELU: ', x.shape)

        x = self.dropout3d(x)
        # ##print('SHAPE AFTER DROPOUT: ', x.shape)

        x = self.pool1(x)
        # ##print('SHAPE AFTER POOL: ', x.shape)

        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool3(x)
        
        # (B, C, T, H, W)->(T, B, C, H, W)
        # ##print('SHAPE BEFORE PERMUTE: ', x.shape)
        # ##print('FTER CONV BEFORE PERMUTE: ', x)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # ##print('SHAPE AFTER PERMUTE: ', x.shape)

        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)
        # ##print('SHAPE BEFORE GRU: ', x.shape)
        # ##print('BEFORE GRU ', x)
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        
        x, h = self.gru1(x)        
        x = self.dropout(x)
        x, h = self.gru2(x)   
        x = self.dropout(x)
        #print('AFTER GRU: ', x.shape)        
        x = self.FC(x)
        #print('AFTER FC: ', x.shape)
        x = x.permute(1, 0, 2).contiguous()
        # ##print('SHPAE AFTER MyModel: ', x.shape)
        # ##print('AFTER MyModel: ', x)
        return x
        
    