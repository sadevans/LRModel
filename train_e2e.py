import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
from dataset import MyDataset
from datamodule import DataModule
import numpy as np
import time
import torch.optim as optim
import re
import json
# from tensorboardX import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
# from torchsummary import summary
from torch.optim import Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

# from model import Model
from NEW.model.e2e import E2E

def ctc_decode(y):
    result = []
    # ##print(y.shape)
    y = y.argmax(-1)
    # ##print(y)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]

def to_device(m, x):
    """Send tensor into the device of the module.

    Args:
        m (torch.nn.Module): Torch module.
        x (Tensor): Torch tensor.

    Returns:
        Tensor: Torch tensor located in the same place as torch module.

    """
    if isinstance(m, torch.nn.Module):
        device = next(m.parameters()).device
    elif isinstance(m, torch.Tensor):
        device = m.device
    else:
        raise TypeError(
            "Expected torch.nn.Module or torch.tensor, " f"bot got: {type(m)}"
        )
    return x.to(device)


def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()  


def test(model, datamodule, loss):

    
    # model = model.to(device)
    val_dataloader = datamodule.val_dataloader()    
    model.eval()
    # loader = dataset2dataloader(dataset, shuffle=False)
    loss_list = []
    wer = []
    cer = []
    crit = nn.CTCLoss()
    tic = time.time()
    for (i_iter, input) in enumerate(val_dataloader):            
        vid = input.get('vid').cuda()
        txt = input.get('txt').cuda()
        vid_len = input.get('vid_len').cuda()
        txt_len = input.get('txt_len').cuda()
        vid = vid.permute(0, 2, 1, 3, 4)
        y = model(vid)
        
        # loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
        # loss_list.append(loss)
        # # pred_txt = ctc_decode(y)
        
        # truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
        # wer.extend(MyDataset.wer(pred_txt, truth_txt)) 
        # cer.extend(MyDataset.cer(pred_txt, truth_txt))              
        # if(i_iter % 20 == 0):
        #     v = 1.0*(time.time()-tic)/(i_iter+1)
        #     eta = v * (len(val_dataloader)-i_iter) / 3600.0
            
        #     ##print(''.join(101*'-'))                
        #     ##print('{:<50}|{:>50}'.format('predict', 'truth'))
        #     ##print(''.join(101*'-'))                
        #     for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
        #         ##print('{:<50}|{:>50}'.format(predict, truth))                
        #     ##print(''.join(101 *'-'))
        #     ##print('test_iter={},eta={},wer={},cer={}'.format(i_iter,eta,np.array(wer).mean(),np.array(cer).mean()))                
        #     ##print(''.join(101 *'-'))
            
    # return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())



def train(model, datamodule, optimizer, loss_fn, scheduler=None):
    epochs = 100
    train_loss_history = []
    train_wer = []
    model = model.to(device)
    training_dataloader = datamodule.train_dataloader() 
    # ##print(training_dataloader)
    # ##print(len(training_dataloader.dataset))
    tic = time.time()
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        torch.cuda.empty_cache()
        # model.train()
        for it, input in enumerate(training_dataloader):
            
            vid = input.get('vid').to(device)
            txt = input.get('txt').to(device)

            vid_len = input.get('vid_len').to(device)
            txt_len = input.get('txt_len').to(device)
            batch_size, frames, channels, hight, width = vid.shape
            # ##print(batch_size, frames, channels, hight, width)
            batch_size, seq_length = txt.shape
            # vid = vid.permute(0, 2, 1, 3, 4)

            logits = model(vid, show=False)                                          # [Batch, Seq Length, Class]
            #print(logits.shape)
            # targets = torch.tensor([1, 5, 9, 100], device='cuda' if torch.cuda.is_available() else 'cpu')
            # loss = loss_fn(logits, targets)/batch_size

            pred_alignments_for_ctc = logits.permute(1, 0, -1)               # [Seq Length, Batch, Class]
            #print("pred alignments shape: ", pred_alignments_for_ctc.shape)
            txts = [t[t != 0] for t in txt]

            input_length = torch.sum(torch.ones_like(logits[:, :, 0]), dim=1).int()
            label_length = torch.sum(torch.ones_like(txt), dim=1)
            #print(logits.shape)
            loss = loss_fn(pred_alignments_for_ctc.log_softmax(-1), 
                        torch.cat(txts), 
                        # txt,
                        # vid_len,
                        input_length, 
                        txt_len)/batch_size
            # #print(loss)
            loss.backward()
            optimizer.step()
            #print(loss)
            ##print(pred_alignments)
        #     break
        # break
            # ##print('OUTPUT SHAPE: ', pred_alignments.shape)
            # pred_alignments_for_ctc = pred_alignments.permute(1, 0, -1)               # [Seq Length, Batch, Class]
            # txts = [t[t != 0] for t in txt]
            # ##print(pred_alignments.log_softmax(-1).transpose(1, 0).shape, txt.shape, vid_len.shape, txt_len.shape, pred_alignments_for_ctc.shape)
            # pred_alignments_padded = nn.utils.rnn.pad_packed_sequence(pred_alignments)

            # input_length = torch.sum(torch.ones_like(pred_alignments[:, :, 0]), dim=1).int()
            # label_length = torch.sum(torch.ones_like(txt), dim=1)

            # ##print(input_length.shape, label_length.shape)
            # ##print(input_length, vid_len, txt_len, [torch.nonzero(pred_alignments[i]).size(0) for i in range(pred_alignments.shape[0])])


            # ##print(pred_alignments.log_softmax(-1).transpose(1, 0).shape)
            # loss = loss_fn(pred_alignments.log_softmax(-1).transpose(1, 0), 
            #             torch.cat(txts), 
            #             # txt,
            #             # vid_len,
            #             input_length, 
            #             txt_len)
            
           
            # ##print('STEP LOSS: ', loss)
            # loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         ##print("Gradients for parameter", name)
            #         ##print(param.grad)
            #     else: ##print('param.grad is None')
            # optimizer.step()

            if scheduler is not None:
                scheduler.step(loss)

            # tot_iter = it + epoch*len(training_dataloader)
            
            # ##print(pred_alignments)
            pred_txt = ctc_decode(logits)
            #print(pred_txt)
            # ##print(pred_alignments, pred_alignments_for_ctc)
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))
            
            # if(tot_iter % 10 == 0):
            #     v = 1.0*(time.time()-tic)/(tot_iter+1)
            #     eta = (len(training_dataloader)-it)*v/3600.0
                
                # writer.add_scalar('train loss', loss, tot_iter)
                # writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)              
                # ##print(''.join(101*'-'))                
                # ##print('{:<50}|{:>50}'.format('predict', 'truth'))                
                # ##print(''.join(101*'-'))
                
                # for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                #     ##print('{:<50}|{:>50}'.format(predict, truth))
                # ##print(''.join(101*'-'))                
                # ##print('epoch={},tot_iter={},eta={},loss={},train_wer={}'.format(epoch, tot_iter, eta, loss, np.array(train_wer).mean()))
                # ##print('epoch={},tot_iter={},eta={},loss={},train_wer={}'.format(epoch, tot_iter, eta, loss, MyDataset.wer(pred_txt, truth_txt).mean()))
                
                # ##print(''.join(101*'-'))
                





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
    model = E2E("/home/sadevans/space/personal/LRModel/config_ef.yaml", efficient_net_size="B")
    ##print(model)
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-6, weight_decay=1e-5, betas=(0.9, 0.98))


    # optimizer = RMSprop(params=model.parameters(), lr=1e-6, weight_decay=1e-5, momentum= 0.9)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 3, 50, len(datamodule.train_dataloader()))
    # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

    optimizer = Adam(params=model.parameters(), 
                 lr=1e-8,
                 amsgrad=True,)
    
    # optimizer =  Adam(params=model.parameters(), weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-10, verbose=True)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-8)

    ##print('LEN LETTERS:', len(MyDataset.letters))
    # blank=len(MyDataset.letters),
    loss_fn = nn.CTCLoss(zero_infinity=True, reduction='sum')
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = CTCLossWithLengthPenalty(length_penalty_factor=0.5)
    ##print(optimizer)
    train(model, datamodule, optimizer, loss_fn, scheduler)
