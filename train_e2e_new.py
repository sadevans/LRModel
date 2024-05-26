import torch
# import wandb
import torch.nn as nn
import torch.nn.functional as F
from dataset import MyDataset
from datamodule import DataModule
import numpy as np
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch import nn
from torch.optim import Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, LambdaLR
from NEW.model.e2e import E2E
learning_rate = 0.18 
warmup_epochs = 3
decay_epochs = 2.4
decay_rate = 0.97
rms_decay = 0.9
momentum = 0.9
bn_momentum = 0.99
weight_decay = 1e-5
ema_decay = 0.9999
dropout_rate = 0.3
survival_probability = 0.8


def ctc_decode(y):
    result = []
    # ##print(y.shape)
    y = y.argmax(-1)
    # ##print(y)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]


def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return 1e-6 + (learning_rate - 1e-6) * epoch / warmup_epochs
    else:
        return decay_rate ** ((epoch - warmup_epochs) // decay_epochs)

def train(model, datamodule, optimizer, scheduler, loss_fn, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # wandb.init(project="lipreading", config={
    #     "learning_rate": optimizer.param_groups[0]['lr'],
    #     "epochs": epochs,
    #     # Log any additional relevant model or training parameters
    # })

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in datamodule.train_dataloader():
            # vid = batch.get('vid').to(device)
            # txt = batch.get('txt').to(device)

            # vid_len = batch.get('vid_len').to(device)
            # txt_len = batch.get('txt_len').to(device)
            vid, txt, vid_len, txt_len = batch
            print(vid.shape, txt.shape, txt_len.shape, vid_len.shape)
            vid = vid.to(device)
            txt = txt.to(device)

            vid_len = vid_len.to(device)
            txt_len = txt_len.to(device)
            batch_size, frames, channels, hight, width = vid.shape

            optimizer.zero_grad()
            pred_alignments = model(vid)
            print('HERE: ', pred_alignments.shape)
            pred_alignments_for_ctc = pred_alignments.permute(1, 0, -1)               # [Seq Length, Batch, Class]
            txts = [t[t != 0] for t in txt]
            input_length = torch.sum(torch.ones_like(pred_alignments[:, :, 0]), dim=1).int()
            # label_length = torch.sum(torch.ones_like(txt), dim=1)
            loss = loss_fn(pred_alignments_for_ctc.log_softmax(-1), 
                        # torch.cat(txts),
                        txt, 
                        # input_length, 
                        vid_len,
                        txt_len)
            

            # loss = loss.mean()
            # loss = loss/batch_size
            loss.backward()
            print("LOSS: ", loss)
            optimizer.step()

            train_loss += loss.item()
            if scheduler is not None:
                scheduler.step(loss)

            pred_txt = ctc_decode(pred_alignments)
            print(pred_txt)
            truth_txt = [MyDataset.ctc_arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            print(truth_txt)
            # print("CROSS ENTROPY: ", )
        
        avg_train_loss = train_loss / len(datamodule.train_dataloader())
        
        # wandb.log({"train_loss": avg_train_loss, "learning_rate": scheduler.get_last_lr()[0]}, step=epoch) 

        # validate(model, datamodule, loss_fn, epoch, device)
        # scheduler.step() 


def validate(model, datamodule, loss_fn, epoch, device="cuda"):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            vid, txt, vid_len, txt_len = batch
            vid = vid.to(device)
            txt = txt.to(device)

            vid_len = vid_len.to(device)
            txt_len = txt_len.to(device)
            # vid = batch.get('vid').to(device)
            # txt = batch.get('txt').to(device)

            # vid_len = batch.get('vid_len').to(device)
            # txt_len = batch.get('txt_len').to(device)
            logits = model(vid)
            # loss = loss_fn(logits, targets)
            # val_loss += loss.item()

    avg_val_loss = val_loss / len(datamodule.val_dataloader())
    # wandb.log({"val_loss": avg_val_loss}, step=epoch)

    # Additional Validation Metrics:
    # You might want to calculate and log metrics like Word Error Rate (WER) or Sentence Error Rate (SER) 


def test(model, datamodule, loss_fn, device="cuda"):
    model.eval()
    test_loss = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            videos, targets = batch
            videos, targets = videos.to(device), targets.to(device)
            logits = model(videos)
            loss = loss_fn(logits, targets)
            test_loss += loss.item()
        
            # Store predictions and targets for later evaluation
            # (depending on your decoding, you may need to convert logits to text)
            all_predictions.extend(logits.argmax(dim=-1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_test_loss = test_loss / len(datamodule.test_dataloader())
    # wandb.log({"test_loss": avg_test_loss})

    # Calculate and log final evaluation metrics (WER, SER, etc.)


class CustomWarmupDecayScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_epochs=3, warmup_start_lr=1e-6, warmup_end_lr=0.18,
                 decay_epochs=2.4, decay_factor=0.97, last_epoch=-1, verbose=False):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr
        self.decay_epochs = decay_epochs
        self.decay_factor = decay_factor
        super(CustomWarmupDecayScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch, verbose)

    def lr_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            return self.warmup_start_lr + (self.warmup_end_lr - self.warmup_start_lr) * epoch / self.warmup_epochs
        else:
            return self.warmup_end_lr * self.decay_factor**((epoch - self.warmup_epochs) / self.decay_epochs)


if __name__ == "__main__":
    BATCH_SIZE = 100
    datamodule = DataModule(
        "video", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_",
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_train_transcript_lengths_seg24s_0to100_5000units.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_valid_transcript_lengths_seg24s_0to100_5000units.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_valid_transcript_lengths_seg24s_0to100_5000units.csv"
    )

    # train_dataloader = datamodule.train_dataloader()

    # for item in train_dataloader:
    #     # ##print(item)
    #     continue
    

    # model = Model(len(MyDataset.letters)+1)
    model = E2E("/home/sadevans/space/personal/LRModel/config_ef.yaml", num_classes=len(MyDataset.letters)+1, efficient_net_size="B")
    ##print(model)
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-6, weight_decay=1e-5, betas=(0.9, 0.98))


    # optimizer = RMSprop(params=model.parameters(), lr=1e-6, weight_decay=1e-5, momentum= 0.9)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 3, 50, len(datamodule.train_dataloader()))
    # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

    # optimizer = Adam(params=model.parameters(), 
    #              lr=1e-8,
    #              amsgrad=True,)
    optimizer = RMSprop(model.parameters(), lr=1e-8, alpha=rms_decay, momentum=momentum, weight_decay=weight_decay)
    
    # optimizer =  Adam(params=model.parameters(), weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-10, verbose=True)
    # scheduler = CustomWarmupDecayScheduler(optimizer, warmup_epochs=3, warmup_start_lr=1e-6, warmup_end_lr=0.18,
    #                                    decay_epochs=2.4, decay_factor=0.97)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-8)

    scheduler = LambdaLR(optimizer, lr_lambda)
    ##print('LEN LETTERS:', len(MyDataset.letters))
    # blank=len(MyDataset.letters),
    # loss_fn = nn.CTCLoss(zero_infinity=True, reduction='sum')
    loss_fn = nn.CTCLoss(reduction='mean', zero_infinity=True, blank=0).to(device)
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = CTCLossWithLengthPenalty(length_penalty_factor=0.5)
    train(model, datamodule, optimizer, None, loss_fn)
