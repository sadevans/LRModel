import torch
import torch.nn as nn
# from pytorch_trainer import Module, data_loader
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from torch.utils.data import DataLoader

# from src.data.lrw import LRWDatset
from .efficientnet import Conv3DEfficientNetV2
from .transformer import TransformerEncoder
from .temporal import TCN, tcn_init
import numpy as np
import matplotlib.pyplot as plt
from .nll_sequence_loss import NLLSequenceLoss
from scheduler import WarmupCosineScheduler
from torch import nn, optim
from src.data.lrw import LRWDataset

import os
from ema_pytorch import ExponentialMovingAverage


class E2E(LightningModule):
    def __init__(self, config, hparams=None, in_channels=1, augmentations=False, num_classes=34, efficient_net_size="S") :
        # super(E2E, self).__init__()
        super().__init__()
        # print(hparams)

        self.save_hyperparameters(hparams)

        # print(self.hparams)
        self.ema = ExponentialMovingAverage(self.encoder.parameters(), decay=0.995)
        self.in_channels = in_channels
        self.augmentations = augmentations
        # self.num_classes = num_classes
        self.num_classes = self.hparams.words

        self.frontend_3d = Conv3DEfficientNetV2(config, efficient_net_size=efficient_net_size)

        self.transformer_encoder = TransformerEncoder(dropout=0.3)
        self.tcn_block = TCN(dropout=0.3)
        # tcn_init(self.tcn_block)

        self.temporal_avg = nn.AdaptiveAvgPool1d(1)

        self.fc_layer = nn.Linear(463, self.num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.loss = NLLSequenceLoss()

        self.best_val_acc = 0
        self.epoch = 0

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

        # return self.softmax(x)
        return x.log_softmax(dim=1)
    
    def training_step(self, batch, batch_num):
        frames = batch['frames']
        labels = batch['label']
        # print("TRUE LABELS: ", labels)
        output = self.forward(frames)
        loss = self.loss(output, labels.squeeze(1))
        # print("LOSS: ", loss)
        loss = loss.mean()
        acc = accuracy(output, labels)
        # logs = {'train_loss': loss, 'train_acc': acc, 'loss':loss}
        # self.log({'train_loss': loss})
        # lr_ = self.scheduler.get_last_lr()
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)


        # return {'loss': loss, 'acc': acc, 'log': logs}
        return loss

    def validation_step(self, batch, batch_num):
        frames = batch['frames']
        labels = batch['label']
        words = batch['word']
        # print("TRUE LABELS: ", labels)
        # print("WORDS: ", words)
        output = self.forward(frames)
        loss = self.loss(output, labels.squeeze(1))
        acc = accuracy(output, labels)
        sums = torch.sum(output, dim=1)
        _, predicted = sums.max(dim=-1)
        # self.log("val_loss", acc, on_step=False, on_epoch=True, batch_size=self.hparams.bach_size, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        return {
            'val_loss': loss,
            'val_acc': acc,
            'predictions': predicted,
            'labels': labels.squeeze(dim=1),
            'words': words,
            # 'loss':loss
        }
        # return loss

    def validation_end(self, outputs):
        predictions = torch.cat([x['predictions'] for x in outputs]).cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy()
        words = np.concatenate([x['words'] for x in outputs])
        self.confusion_matrix(labels, predictions, words)

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        if self.best_val_acc < avg_acc:
            self.best_val_acc = avg_acc
        logs = {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'best_val_acc': self.best_val_acc
        }
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        self.log("val_acc", avg_acc, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        self.log("best_val_acc", self.best_val_acc, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)


        self.epoch += 1
        return {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'log': logs,
        }
        # return logs
        # return avg_loss

    def on_before_zero_grad(self):
        self.ema.update(self.parameters())

    def confusion_matrix(self, label, prediction, words, normalize=True):
        classes = unique_labels(label, prediction)
        cm = confusion_matrix(prediction, label)
        cmap = plt.cm.Blues
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
        ax.set_ylabel("Label")
        ax.set_xlabel("Predicted")
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title("Word Confusion Matrix")

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        directory = "data/viz/lrw"
        os.makedirs(directory, exist_ok=True)
        path = f"{directory}/cm_seed_{self.hparams.seed}_epoch_{self.epoch}.png"
        plt.savefig(path)
        self.logger.save_file(path)
        plt.clf()
        plt.close()

    def configure_optimizers(self):
        # return optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, betas=(0.))
        optimizer = optim.RMSprop([{"name": "model", "params": self.parameters(), "lr": self.hparams.lr}],  weight_decay=self.hparams.weight_decay, alpha=0.9, momentum= 0.9)
        # optimizer = optim.RMSprop(params=self.parameters(), lr=self.hparams.lr,  weight_decay=self.hparams.weight_decay, alpha=0.9, momentum= 0.9)
        
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-8, factor=0.5, patience=2.4)
        # return [optimizer], [scheduler]
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_acc"}
        scheduler = WarmupCosineScheduler(optimizer, 3, 20, len(self.train_dataloader()))
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # return optimizer
        return [optimizer], [scheduler]


    def train_dataloader(self):
        train_data = LRWDataset(
            path=self.hparams.data,
            num_words=self.hparams.words,
            in_channels=self.in_channels,
            augmentations=self.augmentations,
            # query=self.query,
            estimate_pose=False,
            seed=self.hparams.seed
        )
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_data = LRWDataset(
            path=self.hparams.data,
            num_words=self.hparams.words,
            in_channels=self.in_channels,
            mode='val',
            # query=self.query,
            estimate_pose=False,

            seed=self.hparams.seed
        )
        val_loader = DataLoader(val_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers)
        return val_loader

    def test_dataloader(self):
        test_data = LRWDataset(
            path=self.hparams.data,
            num_words=self.hparams.words,
            in_channels=self.in_channels,
            mode='test',
            # query=self.query,
            estimate_pose=False,

            seed=self.hparams.seed
        )
        test_loader = DataLoader(test_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers)
        return test_loader


def accuracy(output, labels):
    # print(output, labels)
    sums = torch.sum(output, dim=1)
    _, predicted = sums.max(dim=-1)
    correct = (predicted == labels.squeeze(dim=1)).sum().type(torch.FloatTensor)
    return correct / output.shape[0]
    





# if __name__ == "__main__":
#     test_tensor = torch.FloatTensor(np.random.rand(1, 29, 1, 88, 88))
#     model = E2E("/home/sadevans/space/personal/LRModel/config_ef.yaml", efficient_net_size="B")
#     model(test_tensor, show=False)
