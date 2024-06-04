from torch import nn
import torch

class NLLSequenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.NLLLoss()


    def forward(self, pred, target):
        loss = 0.0
        pred = pred.contiguous()

        num_frames = pred.shape[0]
        for i in range(num_frames):
            it = self.criterion(pred[i], torch.tensor(target[i].clone().detach(),dtype=torch.long))
            loss += it
        return loss
