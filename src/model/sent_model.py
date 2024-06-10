import torch
import torch.nn as nn
# from pytorch_trainer import Module, data_loader
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from torch.utils.data import DataLoader
from src.data.lrw import LRWDataset
from ..data.transform_sent import VideoTransform

# from src.data.lrw import LRWDatset
from .efficientnet import Conv3DEfficientNetV2
from .transformer import TransformerEncoder
from .temporal import TCN, tcn_init
import numpy as np
import matplotlib.pyplot as plt
# from .nll_sequence_loss import CTCLoss
from scheduler import WarmupCosineScheduler
from torch import nn, optim
import torch.nn.functional as F

from pyctcdecode import build_ctcdecoder
# from pytorch_ignite.handlers import EMAHandler
import os

from ..data.sentences import MyDataset, MyLRWDataset
from torch.nn.utils.rnn import pad_sequence
from .ctc_beamsearch import CTCBeamSearchDecoder
# from jiwer import wer, cer
import jiwer

# from .ctc_decoder import Decoder
# from ema_pytorch import ExponentialMovingAverage
# from textblob import TextBlob

# def correct_spelling(sentences):
#     corrected_sentences = []
#     for sentence in sentences:
#         blob = TextBlob(sentence)
#         corrected_sentence = str(blob.correct())
#         corrected_sentences.append(corrected_sentence)
#     return corrected_sentences
from textblob import TextBlob
from multiprocessing import Pool

def correct_single_sentence(sentence):
    blob = TextBlob(sentence)
    return str(blob.correct())

def correct_spelling_parallel(sentences, num_processes=None):
    with Pool(processes=num_processes) as pool:  # Use available CPUs by default
        corrected_sentences = pool.map(correct_single_sentence, sentences)
    return corrected_sentences


def ctc_decode(y):
    result = []
    # print(y.shape)
    y = torch.exp(y)
    # print("Y AFTER EXP: ", y)
    # y = y.argmax(dim=-1)
    y = y.argmax(-1)

    # print("LABELS CLASS: ", y)
    # print("SHAPE IN CTC DECODE: ", y.shape)
    # return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]
    return MyLRWDataset.ctc_arr2txt(y, start=1)

def ctc_collate(batch):
    lip_features, transcripts, feat_lengths, label_lengths = zip(*batch)

    max_label_length = max(label_lengths)
    input_seq_length = max_label_length * 2

    lip_features_padded = pad_sequence([feat[:input_seq_length] for feat in lip_features],
                                        batch_first=True, padding_value=0)
    
    transcripts_padded = pad_sequence(transcripts, batch_first=True, padding_value=0)

    return lip_features_padded, transcripts_padded, torch.tensor(feat_lengths), torch.tensor(label_lengths)


class E2E(LightningModule):
    def __init__(self, config, hparams=None, dropout=0.3, in_channels=1, augmentations=False, num_classes=34, efficient_net_size="S") :
        # super(E2E, self).__init__()
        super().__init__()
        self.dropout_rate = dropout
        self.save_hyperparameters(hparams)
        self.in_channels = in_channels
        self.augmentations = augmentations

        
        # self.num_classes = self.hparams.words
        # self.characters = self.train_dataloader.dataset.characters
        # self.characters = MyDataset.characters
        self.characters = ['-']+ MyLRWDataset.characters
        # self.decoder = CTCBeamSearchDecoder(self.characters)
        self.decoder = build_ctcdecoder(labels=MyLRWDataset.characters)
        self.num_classes = len(self.characters)
        print("CLASSES: ", self.num_classes, self.characters)

        self.frontend_3d = Conv3DEfficientNetV2(config, efficient_net_size=efficient_net_size)

        self.transformer_encoder = TransformerEncoder(dropout=self.dropout_rate)
        self.tcn_block = TCN(dropout=self.dropout_rate)
        # tcn_init(self.tcn_block)

        self.temporal_avg = nn.AdaptiveAvgPool1d(1)

        self.fc_layer = nn.Linear(463, self.num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.best_val_wer = 0
        self.best_val_cer = 0
        self.best_val_acc = 0

        self.epoch = 0
        self.sum_batches = 0.0

        # self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.criterion = nn.CTCLoss(zero_infinity=True, reduction='mean', blank=0)

        # self.decoder = GreedyDecoder


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

        # x = x.transpose(1, 0)
        # x = self.temporal_avg(x)
        # x = x.transpose(1, 0)

        # else: x = x.transpose(2,1)
        x = x.transpose(2,1)
        # x = x.squeeze()
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
        # return x

    
    def training_step(self, batch, batch_num):
        frames, labels, lengths, labels_lengths = batch
        output = self.forward(frames)
        output_ctc = output.transpose(0, 1)
        input_lengths = torch.full((output_ctc.size(1),), output_ctc.size(0), dtype=torch.long)
        loss = self.criterion(output_ctc, labels, input_lengths, labels_lengths)

        # print("LOSS: ", loss)
        pred_txt = [ctc_decode(_) for _ in output]
        pred_txt = correct_spelling_parallel(pred_txt, num_processes=3)
        # pred_txt = [self.decoder.decode(torch.exp(_).cpu().detach().numpy()) for _ in output]

        # pred_txt = self.decoder.decode_beam(output, lengths)

        truth_txt = [MyLRWDataset.arr2txt(labels[_], start=1) for _ in range(labels.size(0))]
        print(pred_txt, truth_txt)

        # wer = self.decoder.wer_batch(pred_txt, truth_txt)
        # cer = self.decoder.cer_batch(pred_txt, truth_txt)

        # wer = MyLRWDataset.wer(pred_txt, truth_txt).mean()
        # cer = MyLRWDataset.cer(pred_txt, truth_txt).mean()

        wer = np.array([jiwer.wer(truth_txt_, pred_txt_) for truth_txt_, pred_txt_ in zip(truth_txt, pred_txt)]).mean()
        cer = np.array([jiwer.cer(truth_txt_, pred_txt_) for truth_txt_, pred_txt_ in zip(truth_txt, pred_txt)]).mean()

        # print("WER, CER: ", wer, cer)

        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        self.log("train_wer", wer, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        self.log("train_cer", wer, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        
        self.sum_batches += loss

        return {"loss": loss, "train_loss_step": loss, "train_wer_step": wer, "train_cer_stap": cer}

    def validation_step(self, batch, batch_num):
        frames, labels, lengths, labels_lengths = batch
        output = self.forward(frames)
        output_ctc = output.transpose(0, 1)
        input_lengths = torch.full((output_ctc.size(1),), output_ctc.size(0), dtype=torch.long)
        loss = self.criterion(output_ctc, labels, input_lengths, labels_lengths)

        # print("LOSS: ", loss)
        pred_txt = [ctc_decode(_) for _ in output]
        pred_txt = correct_spelling_parallel(pred_txt, num_processes=3)


        # pred_txt = [self.decoder.decode(torch.exp(_).cpu().detach().numpy()) for _ in output]



        # pred_txt = self.decoder.decode_beam(output, lengths)

        truth_txt = [MyLRWDataset.arr2txt(labels[_], start=1) for _ in range(labels.size(0))]
        print(pred_txt, truth_txt)

        # wer = self.decoder.wer_batch(pred_txt, truth_txt)
        # cer = self.decoder.cer_batch(pred_txt, truth_txt)
        # wer = MyLRWDataset.wer(pred_txt, truth_txt).mean()
        # cer = MyLRWDataset.cer(pred_txt, truth_txt).mean()

        wer = np.array([jiwer.wer(truth_txt_, pred_txt_) for truth_txt_, pred_txt_ in zip(truth_txt, pred_txt)]).mean()
        cer = np.array([jiwer.cer(truth_txt_, pred_txt_) for truth_txt_, pred_txt_ in zip(truth_txt, pred_txt)]).mean()
        # print("WER, CER: ", wer, cer)

        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        self.log("val_wer", wer, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        self.log("val_cer", cer, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        
        return {
            'val_loss': loss,
            'val_wer': wer,
            'val_cer': cer
            # 'predictions': predicted,
            # 'labels': labels.squeeze(dim=1),
            # 'words': words,
        }
    

    def test_step(self, batch, batch_idx):
        frames, labels, lengths, labels_lengths = batch
        output = self.forward(frames)
        output_ctc = output.transpose(0, 1)
        input_lengths = torch.full((output_ctc.size(1),), output_ctc.size(0), dtype=torch.long)
        loss = self.criterion(output_ctc, labels, input_lengths, labels_lengths)

        # print("LOSS: ", loss)
        pred_txt = [ctc_decode(_) for _ in output]
        pred_txt = correct_spelling_parallel(pred_txt, num_processes=3)


        # pred_txt = [self.decoder.decode(torch.exp(_).cpu().detach().numpy()) for _ in output]



        # pred_txt = self.decoder.decode_beam(output, lengths)

        truth_txt = [MyLRWDataset.arr2txt(labels[_], start=1) for _ in range(labels.size(0))]
        print(pred_txt, truth_txt)

        # wer = self.decoder.wer_batch(pred_txt, truth_txt)
        # cer = self.decoder.cer_batch(pred_txt, truth_txt)
        # wer = MyLRWDataset.wer(pred_txt, truth_txt).mean()
        # cer = MyLRWDataset.cer(pred_txt, truth_txt).mean()

        wer = np.array([jiwer.wer(truth_txt_, pred_txt_) for truth_txt_, pred_txt_ in zip(truth_txt, pred_txt)]).mean()
        cer = np.array([jiwer.cer(truth_txt_, pred_txt_) for truth_txt_, pred_txt_ in zip(truth_txt, pred_txt)]).mean()
        print("WER, CER: ", wer, cer)

        # self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        # self.log("test_wer", wer, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        # self.log("test_cer", cer, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        
        return {"test_loss": loss, "test_wer": wer, "test_cer": cer}
    

    def configure_optimizers(self):
        optimizer = optim.AdamW([{"name": "model", "params": self.parameters(), "lr": self.hparams.lr}], weight_decay=self.hparams.weight_decay, betas=(0.9, 0.98))
        
        # optimizer = optim.AdamW([{"name": "model", "params": self.parameters(), "lr": self.hparams.lr}], weight_decay=0.02)
        
        scheduler = WarmupCosineScheduler(optimizer, 3, self.hparams.epochs, len(self.train_dataloader()))
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


    def dataloader_(self, dataset, shuffle=False,sampler=None, collate_fn=None):
        return DataLoader(
            dataset,
            pin_memory=True,
            batch_size = self.hparams.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
    

    # def train_dataloader(self):
    #     train_ds = MyDataset(
    #         root_dir=self.hparams.root_dir,
    #         label_path=os.path.join(
    #             self.hparams.root_dir, self.hparams.label_dir, self.hparams.train_file
    #         ),
    #         subset="train",
    #         modality=self.hparams.modality,
    #         video_transform=VideoTransform("train"),
    #     )
    #     return self.dataloader_(train_ds, shuffle=True,collate_fn=ctc_collate)
    #     # return self.dataloader_(train_ds, shuffle=True,collate_fn=None)

    

    # def val_dataloader(self):
    #     val_ds = MyDataset(
    #         root_dir=self.hparams.root_dir,
    #         label_path=os.path.join(self.hparams.root_dir, self.hparams.label_dir, self.hparams.val_file),
    #         subset="val",
    #         modality=self.hparams.modality,
    #         video_transform=VideoTransform("val"),
    #     )
    #     return self.dataloader_(val_ds, collate_fn=ctc_collate)
    #     # return self.dataloader_(val_ds, collate_fn=None)


    # def test_dataloader(self):
    #     dataset = MyDataset(
    #         root_dir=self.hparams.root_dir,
    #         label_path=os.path.join(self.hparams.root_dir, self.hparams.label_dir, self.hparams.test_file),
    #         subset="test",
    #         modality=self.hparams.modality,
    #         video_transform=VideoTransform("test"),
    #     )
    #     test_dataloader = DataLoader(dataset, batch_size=None)
    #     return test_dataloader
    def train_dataloader(self):
        train_data = MyLRWDataset(
            path=self.hparams.data,
            num_words=self.hparams.words,
            in_channels=self.in_channels,
            augmentations=self.augmentations,
            estimate_pose=False,
            seed=self.hparams.seed
        )
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.hparams.batch_size, \
                                  num_workers=self.hparams.workers, pin_memory=True, collate_fn=ctc_collate)
        return train_loader

    def val_dataloader(self):
        val_data = MyLRWDataset(
            path=self.hparams.data,
            num_words=self.hparams.words,
            in_channels=self.in_channels,
            mode='val',
            estimate_pose=False,
            seed=self.hparams.seed
        )
        val_loader = DataLoader(val_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers, collate_fn=ctc_collate)
        return val_loader

    def test_dataloader(self):
        test_data = MyLRWDataset(
            path=self.hparams.data,
            num_words=self.hparams.words,
            in_channels=self.in_channels,
            mode='test',
            estimate_pose=False,
            seed=self.hparams.seed
        )
        test_loader = DataLoader(test_data, shuffle=True, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers, collate_fn=ctc_collate)

        # test_loader = DataLoader(test_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers)
        return test_loader



def accuracy(predictions, labels):
    preds = torch.exp(predictions)
    preds_ = torch.argmax(preds, dim=1)
    correct = (preds_ == labels).sum().item()
    accuracy = correct / labels.shape[0]
    return accuracy
