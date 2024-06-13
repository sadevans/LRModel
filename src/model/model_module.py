import torch
import torch.nn as nn
# from pytorch_trainer import Module, data_loader
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from torch.utils.data import DataLoader
from src.data.lrw import LRWDataset
from sklearn.metrics import f1_score, precision_score, recall_score

# from src.data.lrw import LRWDatset
from .efficientnet import Conv3DEfficientNetV2
from .transformer import TransformerEncoder
from .temporal import TCN, tcn_init
import numpy as np
import matplotlib.pyplot as plt
from .nll_sequence_loss import NLLSequenceLoss
from scheduler import WarmupCosineScheduler
from torch import nn, optim
import torch.nn.functional as F
# from pytorch_ignite.handlers import EMAHandler
import os
import gc
import copy
from copy import deepcopy
from .e2e import E2E

class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)




class ModelModule(LightningModule):
    def __init__(self, config, hparams=None, dropout=0.3, in_channels=1, augmentations=False, num_classes=34, efficient_net_size="T", use_ema=True) :
        torch.cuda.empty_cache()
        gc.collect()
        super().__init__()
        self.dropout_rate = dropout
        self.save_hyperparameters(hparams)

        self.in_channels = in_channels
        self.augmentations = augmentations
        self.num_classes = self.hparams.words

        self.model = E2E( "/home/sadevans/space/LRModel/config_ef.yaml", dropout=self.dropout_rate, in_channels=self.in_channels, \
                         augmentations=False, num_classes=self.num_classes, efficient_net_size="S")
        
        self.model_ema = EMA(self.model, decay=0.999) if use_ema else None

        self.best_val_acc = 0
        self.epoch = 0
        self.sum_batches = 0.0

        self.criterion = nn.NLLLoss()

        self.test_f1 = []
        self.test_precision = []
        self.test_recall = []
        self.test_acc = []
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")


    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")


    def shared_step(self, batch, mode):
        frames = batch['frames']
        labels = batch['label']
        output = self.model(frames) if self.training or self.model_ema is None else self.model_ema.module(frames)
        loss = self.criterion(output, labels.squeeze(1))
        acc = accuracy(output, labels.squeeze(1))

        if mode == "val":
            self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
            self.log("val_acc", acc, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
            return {
            'val_loss': loss,
            'val_acc': acc,
            'predictions': torch.argmax(torch.exp(output), dim=1),
            'labels': labels.squeeze(dim=1),
            'words': batch['word'],
            }
            
        elif mode == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
            self.log("train_acc", acc, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
            return {"loss": loss, "train_loss_step": loss, "train_acc_step": acc}


    def on_before_backward(self, loss):
        if self.model_ema:
            self.model_ema.update(self.model)

    
    def test_step(self, batch, batch_idx):
        frames = batch['frames']
        labels = batch['label']
        words = batch['word']
        output = self.model(frames)

        loss = self.criterion(output, labels.squeeze(1))
        acc = accuracy(output, labels.squeeze(1))
        return {
            'test_loss': loss,
            'test_acc': acc,
            'predictions': torch.argmax(torch.exp(output), dim=1),
            'labels': labels.squeeze(dim=1),
            'words': batch['word'],
            }
    
    def test_epoch_end(self, outputs):
        predictions = torch.cat([x['predictions'] for x in outputs]).cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy()
        words = np.concatenate([x['words'] for x in outputs])
        acc = np.array([x['test_acc'] for x in outputs])

        f1_score = multiclass_f1(labels, predictions)
        precision = multiclass_precision(labels, predictions)
        recall = multiclass_recall(labels, predictions)
    
        print(f"AVERAGE ACCURACY: {acc.mean()}")

        print(f"F1 score: {f1_score:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")


    def validation_epoch_end(self, outputs):
        predictions = torch.cat([x['predictions'] for x in outputs]).cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy()
        words = np.concatenate([x['words'] for x in outputs])
        # acc = np.array([x['test_acc'] for x in outputs])
        self.confusion_matrix(labels, predictions, words)
        avg_acc = torch.FloatTensor([x['val_acc'] for x in outputs]).mean()
        avg_loss = torch.FloatTensor([x['val_loss'] for x in outputs]).mean()

        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        if self.best_val_acc < avg_acc:
            self.best_val_acc = avg_acc

        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        self.log("val_acc", avg_acc, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        self.log("best_val_acc", self.best_val_acc, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)

        return {
                'val_loss': avg_loss,
                'val_acc': avg_acc
                }


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
        # self.logger.save_file(path)
        plt.clf()
        plt.close()

    def configure_optimizers(self):
        # optimizer = optim.RMSprop([{"name": "model", "params": self.parameters(), "lr": self.hparams.lr}],  weight_decay=self.hparams.weight_decay, alpha=0.9, momentum= 0.9)
        # optimizer = optim.RMSprop(params=self.parameters(), lr=self.hparams.lr,  weight_decay=self.hparams.weight_decay, alpha=0.9, momentum= 0.9)
        # optimizer = optim.Adam([{"name": "model", "params": self.parameters(), "lr": self.hparams.lr}], weight_decay=self.hparams.weight_decay, betas=(0.9, 0.98))
        
        optimizer = optim.AdamW([{"name": "model", "params": self.parameters(), "lr": self.hparams.lr}], weight_decay=self.hparams.weight_decay, betas=(0.9, 0.98))
        
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-8, factor=0.5, patience=2.4)
        # return [optimizer], [scheduler]
        scheduler = WarmupCosineScheduler(optimizer, self.hparams.warmup_epochs, 20, len(self.train_dataloader()))
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # return [optimizer]
        return [optimizer], [scheduler]


    def train_dataloader(self):
        train_data = LRWDataset(
            path=self.hparams.data,
            num_words=self.hparams.words,
            in_channels=self.in_channels,
            augmentations=self.augmentations,
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
            estimate_pose=False,
            seed=self.hparams.seed
        )
        test_loader = DataLoader(test_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers)
        return test_loader

def accuracy(predictions, labels):
    # print("PREDICTIONS: ", predictions, predictions.shape)
    preds = torch.exp(predictions)
    preds_ = torch.argmax(preds, dim=1)
    # print("preds_: ", preds_, preds_.shape)
    # print("LABELS: ", labels)
    # print("CORRECT: ",  preds_ == labels)
    correct = (preds_ == labels).sum().item()
    # print("CORRECT NUM: ", correct)
    accuracy = correct / labels.shape[0]
    return accuracy  

def multiclass_f1(labels, predictions, average='weighted'):
    """
    Compute the F1 score for multiclass classification.

    Parameters:
    labels (array-like): Ground truth labels
    predictions (array-like): Predicted labels
    average (str, optional): Method to compute the F1 score. Can be 'macro', 'weighted' or 'none'.

    Returns:
    f1 (float): F1 score
    """

    return f1_score(labels, predictions, average=average)

def multiclass_precision(labels, predictions, average='weighted'):
    """
    Compute the Precision score for multiclass classification.

    Parameters:
    labels (array-like): Ground truth labels
    predictions (array-like): Predicted labels
    average (str, optional): Method to compute the Precision score. Can be 'macro', 'weighted' or 'none'.

    Returns:
    precision (float): Precision score
    """
    # preds = torch.exp(predictions)
    # preds_ = torch.argmax(preds, dim=1)
    return precision_score(labels, predictions, average=average)

def multiclass_recall(labels, predictions, average='weighted'):
    """
    Compute the Recall score for multiclass classification.

    Parameters:
    labels (array-like): Ground truth labels
    predictions (array-like): Predicted labels
    average (str, optional): Method to compute the Recall score. Can be 'macro', 'weighted' or 'none'.

    Returns:
    recall (float): Recall score
    """

    return recall_score(labels, predictions, average=average)