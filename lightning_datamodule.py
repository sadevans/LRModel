import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.data.lrw import LRWDataset


class DataModule(LightningDataModule):
    def __init__(self, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)


    def train_dataloader(self):
        train_data = LRWDataset(
            path=self.hparams.data,
            num_words=self.hparams.words,
            in_channels=self.hparams.in_channels,
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
            in_channels=self.hparams.in_channels,
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
            in_channels=self.hparams.in_channels,
            mode='test',
            # query=self.query,
            estimate_pose=False,

            seed=self.hparams.seed
        )
        test_loader = DataLoader(test_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers)
        return test_loader
