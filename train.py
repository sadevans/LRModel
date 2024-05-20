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
from torchsummary import summary
from torch.optim import Adam

from model import Model

class CustomCTCFunction(Function):
    @staticmethod
    def forward(
        ctx,
        log_prob,
        targets,
        input_lengths,
        target_lengths,
        blank,
        zero_infinity,
    ):
        with torch.enable_grad():
            log_prob.requires_grad_()
            loss = F.ctc_loss(log_prob,
                targets,
                input_lengths,
                target_lengths,
                blank,
                reduction="none",
                zero_infinity=zero_infinity
            )

            print('LOSS: ', loss)
            ctx.save_for_backward(
                log_prob,
                loss
            )
        ctx.save_grad_input = None
        # for i, l in enumerate(loss):
        #     # print(l)e
        #     if l.item() == float('inf'):
        #         print(i)
        return loss.clone()

    @staticmethod
    def backward(ctx, grad_output):
        log_prob, loss = (
            ctx.saved_tensors
        )

        if ctx.save_grad_input is None:
            ctx.save_grad_input = torch.autograd.grad(loss, [log_prob], loss.new_ones(*loss.shape))[0]

        gradout = grad_output
        grad_input = ctx.save_grad_input.clone()
        grad_input.subtract_(log_prob.exp()).mul_(gradout.unsqueeze(0).unsqueeze(-1))

        return grad_input, None, None, None, None, None
    
custom_ctc_fn = CustomCTCFunction.apply


def custom_ctc_loss(
    log_prob,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="sum",
    zero_infinity=False,
):
    """The custom ctc loss. ``log_prob`` should be log probability, but we do not need applying ``log_softmax`` before ctc loss or requiring ``log_prob.exp().sum(dim=-1) == 1``.

    Parameters:
        log_prob (T, N, C): C = number of characters in alphabet including blank
                            T = input length
                            N = batch size
                            log probability of the outputs (e.g. torch.log_softmax of logits)
        targets (N, S): S = maximum number of characters in target sequences
        input_lengths (N): lengths of log_prob
        target_lengths (N): lengths of targets
        blank (int): index of blank tokens (default 0)
        reduction (str): reduction methods applied to the output. 'none' | 'mean' | 'sum'
        zero_infinity (bool): if true imputer loss will zero out infinities.
                              infinities mostly occur when it is impossible to generate
                              target sequences using input sequences
                              (e.g. input sequences are shorter than target sequences)
    """

    loss = custom_ctc_fn(
        log_prob,
        targets,
        input_lengths,
        target_lengths,
        blank,
        zero_infinity,
    )

    if zero_infinity:
        inf = float("inf")
        loss = torch.where(loss == inf, loss.new_zeros(1), loss)

    if reduction == "mean":
        print(loss)
        target_length = target_lengths.to(loss).clamp(min=1)

        return (loss / target_length).mean()

    elif reduction == "sum":
        # print(loss)
        # print(loss.sum())
        # print(loss.index('inf'))
        # for i, l in enumerate(loss):
        #     # print(l)e
        #     if l.item() == float('inf'):
        #         print(i)
        # print(targets.shape, loss.shape)
        return loss.sum()

    elif reduction == "none":
        return loss

    else:
        raise ValueError(
            f"Supported reduction modes are: mean, sum, none; got {reduction}"
        )

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

def ctc_decode(y):
    result = []
    y = y.argmax(-1)
    # print(y.shape)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]


# def test(model, datamodule, optimizer, loss_fn):



def train(model, datamodule, optimizer, loss_fn):
    epochs = 1000
    train_loss_history = []
    train_wer = []
    model = model.to(device)
    training_dataloader = datamodule.train_dataloader() 
    print(training_dataloader)
    print(len(training_dataloader.dataset))
    tic = time.time()
    for epoch in range(1, epochs+1):
        train_loss = 0.0
        val_loss = 0.0
        torch.cuda.empty_cache()
        model.train()
        for it, input in enumerate(training_dataloader):
            vid = input.get('vid').to(device)
            # vid = vid.permute(0, 2, 1, 3, 4)
            # txt = input.get('txt').to(device).squeeze()
            txt = input.get('txt').to(device)

            # print(txt, txt.shape)
            break
        # break
            vid_len = input.get('vid_len').to(device)
            txt_len = input.get('txt_len').to(device)
            batch_size, frames, channels, hight, width = vid.shape
            # print(batch_size, frames, channels, hight, width)
            batch_size, seq_length = txt.shape
            
        #     videos = videos.to(device)
        #     alignments = alignments.to(device)
            
            optimizer.zero_grad()
            vid = vid.permute(0, 2, 1, 3, 4)
            pred_alignments = model(vid)                                          # [Batch, Seq Length, Class]
            # print('OUTPUT SHAPE: ', pred_alignments.shape)
            pred_alignments_for_ctc = pred_alignments.permute(1, 0, -1)               # [Seq Length, Batch, Class]
            logits = pred_alignments_for_ctc.log_softmax(2)
            # print(pred_alignments_for_ctc)
            # print('OUTPUT SHAPE AFTER PERMUTE: ', pred_alignments_for_ctc.shape)
            # print(pred_txt)
            txts = [t[t != -1] for t in txt]
            print(txt_len)
            loss = loss_fn(logits, 
                        # torch.cat(txts), 
                        txt,
                        vid_len, 
                        txt_len, ).sum()
            # print('STEP LOSS: ', loss)
            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print("Gradients for parameter", name)
            #         print(param.grad)
            #     else: print('param.grad is None')
            optimizer.step()
            tot_iter = it + epoch*len(training_dataloader)
            
            pred_txt = ctc_decode(pred_alignments_for_ctc)
            # print(pred_txt)
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))
            
            if(tot_iter % 10 == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(training_dataloader)-it)*v/3600.0
                
                # writer.add_scalar('train loss', loss, tot_iter)
                # writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)              
                print(''.join(101*'-'))                
                print('{:<50}|{:>50}'.format('predict', 'truth'))                
                print(''.join(101*'-'))
                
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101*'-'))                
                print('epoch={},tot_iter={},eta={},loss={},train_wer={}'.format(epoch, tot_iter, eta, loss, np.array(train_wer).mean()))
                print(''.join(101*'-'))
                
            # if(tot_iter % 25 == 0):                
            #     (loss, wer, cer) = test(model)
            #     print('i_iter={},lr={},loss={},wer={},cer={}'
            #         .format(tot_iter,show_lr(optimizer),loss,wer,cer))
            #     # writer.add_scalar('val loss', loss, tot_iter)                    
            #     # writer.add_scalar('wer', wer, tot_iter)
            #     # writer.add_scalar('cer', cer, tot_iter)
            #     # savename = '{}_loss_{}_wer_{}_cer_{}.pt'.format(opt.save_prefix, loss, wer, cer)
            #     # (path, name) = os.path.split(savename)
            #     # if(not os.path.exists(path)): os.makedirs(path)
            #     # torch.save(model.state_dict(), savename)
            #     if(not True):
            #         exit()



        #     tot_iter = it + epoch*len(training_dataloader)

        #     train_loss += loss.item()*batch_size
        # print('EPOCH LOSS: ', train_loss/len(training_dataloader.dataset))
        # train_loss_history.append(train_loss/len(training_dataloader.dataset))




if __name__ == "__main__":
    BATCH_SIZE = 100
    datamodule = DataModule(
        "video", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_",
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_train_transcript_lengths_seg24s_0to100_5000units.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_val_transcript_lengths_seg24s_0to100_5000units.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_val_transcript_lengths_seg24s_0to100_5000units.csv"
    )
    model = Model(len(MyDataset.letters)+1)
    print(model)
    optimizer = Adam(params=model.parameters(), 
                 lr=0.00000001,
                 amsgrad=True)
    print('LEN LETTERS:', len(MyDataset.letters))
    loss_fn = nn.CTCLoss(blank=len(MyDataset.letters),zero_infinity=True)
    print(optimizer)
    train(model, datamodule, optimizer, loss_fn)
