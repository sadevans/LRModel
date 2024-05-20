import torch 
import torch.nn as nn
class Permute(nn.Module):
    def __init__(self, *args: tuple):
        super(Permute, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.permute(self.shape)
    
class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=128, kernel_size=3, padding='same')
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d((1, 2, 2))

        self.conv2 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d((1, 2, 2))

        self.conv3 = nn.Conv3d(in_channels=256, out_channels=75, kernel_size=3, padding='same')
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool3d((1, 2, 2))

        # self.permute = nn.Permute(0, 2, 1, 3, 4)
        self.flatten = nn.Flatten(2, -1)

        self.lstm1 = nn.LSTM(input_size=9075, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        self.linear = nn.Linear(512, vocab_size)

    def forward(self, x):
        # print(x.shape, len(x.shape))
        # print(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        # print(x.size())
        x = x.permute(0, 2, 1, 3, 4)
        # x = x.view(x.size(0), 1, -1)
        
        # x = x.view(batch_size, seq_len, -1)
        x = self.flatten(x)
        batch_size, seq_len, num_features = x.size()
        # print(x.size())
        x = x.view(batch_size, seq_len, -1)

        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()

        # print('X SHAPE: ', x.shape)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.linear(x)

        return x

    # def forward(self, x):
    #     print(x.shape, len(x.shape))
    #     print(x)
    #     # x = x.view(x.shape[1], x.shape[2], x.shape[3], x.shape[4],x.shape[5])
    #     # print(x.shape)

    #     return self.model(x)