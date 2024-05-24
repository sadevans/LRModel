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



def pad(samples, pad_val=0.0):
    lengths = [s.shape[0] for s in samples]
    # print('LENGTHS: ', lengths)
    # print('LENGTHS: ', [s.shape[0] for s in samples])
    
    max_size = max(lengths)
    # print('MAX SIZE : ', max_size)
    # if pad_val == -1:
    #     print('HERE: ', samples, len(samples), samples[0].shape, samples[0].shape[1:])

    # else:
    #     print("VID SHAPE ", len(samples), samples[0].shape, samples[0].shape[1:])

    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)

    for i, sample in enumerate(samples):
        # print('LEN SAMPLE: ', len(sample))
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    # if len(samples[0].shape) == 1:
    #     print('IN TXT COLLATED PAD: ',  collated_batch.shape)
        # collated_batch = collated_batch.unsqueeze(1)  # targets
        # print('IN TXT COLLATED PAD AFTER: ',  collated_batch.shape)

    # elif len(samples[0].shape) == 2:
    #     pass  # collated_batch: [B, T, 1]s
    # elif len(samples[0].shape) == 4:
    #     pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        # pad_val = -1 if data_type == "txt" else 0.0
        pad_val = 0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        # print('SAMPLE LENGTHS: ', torch.tensor(sample_lengths), torch.tensor(np.array(sample_lengths)))
        batch_out[data_type] = c_batch
        batch_out[data_type + "_len"] = torch.tensor(np.array(sample_lengths))

    # print('BATCH: ', batch_out)
    return batch_out


class DataModule:
    def __init__(self, modality, root_dir, train_file, val_file, test_file, label_dir='labels'):
        # self.letters = [' ', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', \
        #        'у', 'ф', 'ц', 'х', 'ш', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']
        self.letters = [char for char in ' абвгдежзийклмнопрстуфхцчшщъыьэюя']
        # self.crg = cfg
        # self.cfg.gpus = torch.cuda.device_count()
        # self.total_gpus = self.cfg.gpus * self.cfg.trainer.num_nodes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modality = modality
        self.root_dir = root_dir
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.label_dir = label_dir

        self.batch_size = 4
        self.total_gpus =  torch.cuda.device_count()


    def dataloader_(self, dataset, shuffle=False,sampler=None, collate_fn=None):
        return DataLoader(
            dataset,
            # num_workers=12,
            pin_memory=True,
            batch_size = self.batch_size,
            shuffle=shuffle,
            # batch_sampler=sampler,
            collate_fn=collate_fn,
        )
    
    def train_dataloader(self):
        # ds_args = self.cfg.data.dataset
        train_ds = MyDataset(
            root_dir=self.root_dir,
            label_path=os.path.join(
                self.root_dir, self.label_dir, self.train_file
            ),
            subset="train",
            modality=self.modality,
            # audio_transform=AudioTransform("train"),
            video_transform=VideoTransform("train"),
        )
        # sampler = ByFrameCountSampler(train_ds, self.batch_size)
        # if self.total_gpus > 1:
        #     sampler = DistributedSamplerWrapper(sampler)
        # else:
        #     sampler = RandomSamplerWrapper(sampler)
        return self.dataloader_(train_ds, shuffle=True,collate_fn=collate_pad)
        # return self.dataloader_(train_ds, shuffle=True,collate_fn=None)

    

    def val_dataloader(self):
        val_ds = MyDataset(
            root_dir=self.root_dir,
            label_path=os.path.join(self.root_dir, self.label_dir, self.val_file),
            subset="val",
            modality=self.modality,
            # audio_transform=AudioTransform("val"),
            video_transform=VideoTransform("val"),
        )
        # sampler = ByFrameCountSampler(
        #     val_ds, self.batch_size, shuffle=False
        # )
        # if self.total_gpus > 1:
        #     sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
        return self.dataloader_(val_ds, collate_fn=collate_pad)
        # return self.dataloader_(val_ds, collate_fn=None)


    def test_dataloader(self):
        dataset = MyDataset(
            root_dir=self.root_dir,
            label_path=os.path.join(self.root_dir, self.label_dir, self.test_file),
            subset="test",
            modality=self.modality,
            # audio_transform=AudioTransform(
            #     "test", snr_target=self.snr_target
            # ),
            video_transform=VideoTransform("test"),
        )
        test_dataloader = DataLoader(dataset, batch_size=None)
        return test_dataloader


if __name__ == "__main__":

    datamodule = DataModule(
        "video", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_",
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_train_transcript_lengths_seg24s_0to100_5000units.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_val_transcript_lengths_seg24s_0to100_5000units.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_val_transcript_lengths_seg24s_0to100_5000units.csv"
    )

    loader = datamodule.train_dataloader()
    for (i_iter, input) in enumerate(loader):
        print(input)
