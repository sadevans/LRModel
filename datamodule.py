import os

import torch

from dataset import MyDataset

from transforms import AudioTransform, VideoTransform

import numpy as np
import glob
import time
import cv2
import os
# from cvtransforms import *
import torch
import glob
import re
import copy
import json
import random
import editdistance
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence



def pad(samples, pad_val=0.0):
    lengths = [s.shape[0] for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        # pad_val = -1 if data_type == "txt" else 0.0
        pad_val = 0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type] = c_batch
        batch_out[data_type + "_len"] = torch.tensor(np.array(sample_lengths))

    return batch_out
    

def ctc_collate(batch):
    '''
    Stack samples into CTC style inputs.
    Modified based on default_collate() in PyTorch.
    By Yuan-Hang Zhang.
    '''
    xs, ys, lens = zip(*batch)
    max_len = max(lens)
    y = []
    for sub in ys:y.append(sub)
    y = pad_sequence(y, batch_first=True, padding_value=0)
    lengths = torch.IntTensor(lens)
    y_lengths = torch.IntTensor([len(label) for label in ys])
    x = pad_sequence(xs, batch_first=True, padding_value=0)
    x = x.narrow(1, 0, max_len)
    return x, y, lengths, y_lengths


class DataModule(pl.LightningDataModule):
    def __init__(self, modality, root_dir, train_file, val_file, test_file, label_dir='labels'):
        # self.letters = [char for char in ' абвгдежзийклмнопрстуфхцчшщъыьэюя']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modality = modality
        self.root_dir = root_dir
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.label_dir = label_dir

        self.batch_size = 80
        self.total_gpus =  torch.cuda.device_count()


    def dataloader_(self, dataset, shuffle=False,sampler=None, collate_fn=None):
        return DataLoader(
            dataset,
            pin_memory=True,
            batch_size = self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
    

    def train_dataloader(self):
        train_ds = MyDataset(
            root_dir=self.root_dir,
            label_path=os.path.join(
                self.root_dir, self.label_dir, self.train_file
            ),
            subset="train",
            modality=self.modality,
            video_transform=VideoTransform("train"),
        )
        # return self.dataloader_(train_ds, shuffle=True,collate_fn=ctc_collate)
        return self.dataloader_(train_ds, shuffle=True,collate_fn=None)

    

    def val_dataloader(self):
        val_ds = MyDataset(
            root_dir=self.root_dir,
            label_path=os.path.join(self.root_dir, self.label_dir, self.val_file),
            subset="val",
            modality=self.modality,
            video_transform=VideoTransform("val"),
        )
        # return self.dataloader_(val_ds, collate_fn=ctc_collate)
        return self.dataloader_(val_ds, collate_fn=None)


    def test_dataloader(self):
        dataset = MyDataset(
            root_dir=self.root_dir,
            label_path=os.path.join(self.root_dir, self.label_dir, self.test_file),
            subset="test",
            modality=self.modality,
            video_transform=VideoTransform("test"),
        )
        test_dataloader = DataLoader(dataset, batch_size=None)
        return test_dataloader


class LipreadingDataModule(pl.LightningDataModule):
    def __init__(self, modality, root, train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=4):
        super().__init__()
        datamodule = DataModule(
        modality, 
        root,
        train_dataset, 
        val_dataset, 
        test_dataset
    )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return datamodule.train_dataloader()  # pin_memory for faster GPU transfer

    def val_dataloader(self):
        return datamodule.val_dataloader()

    def test_dataloader(self):
        return datamodule.test_dataloader()


if __name__ == "__main__":

    datamodule = DataModule(
        "video", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_",
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_train_transcript_lengths_seg24s_0to100_5000units.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_val_transcript_lengths_seg24s_0to100_5000units.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_val_transcript_lengths_seg24s_0to100_5000units.csv"
    )

    loader = datamodule.train_dataloader()
    # for (i_iter, input) in enumerate(loader):
    #     #print(input)
