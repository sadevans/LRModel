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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

            ##print('LOSS: ', loss)
            ctx.save_for_backward(
                log_prob,
                loss
            )
        ctx.save_grad_input = None
        # for i, l in enumerate(loss):
        #     # ##print(l)e
        #     if l.item() == float('inf'):
        #         ##print(i)
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
        ##print(loss)
        target_length = target_lengths.to(loss).clamp(min=1)

        return (loss / target_length).mean()

    elif reduction == "sum":
        # ##print(loss)
        # ##print(loss.sum())
        # ##print(loss.index('inf'))
        # for i, l in enumerate(loss):
        #     # ##print(l)e
        #     if l.item() == float('inf'):
        #         ##print(i)
        # ##print(targets.shape, loss.shape)
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
    # ##print(y.shape)
    y = y.argmax(-1)
    # ##print(y)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]

class CTCLossWithLengthPenalty(nn.Module):
    def __init__(self, blank=0, length_penalty_factor=1.0):
        super(CTCLossWithLengthPenalty, self).__init__()
        self.blank = blank
        self.length_penalty_factor = length_penalty_factor

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Calculates the CTC loss with length penalty.

        Args:
            log_probs: Log probabilities of the predicted output (BATCH, SEQUENCE, CLASS).
            targets: Target labels (BATCH, SEQUENCE).
            input_lengths: Length of the input sequence (BATCH).
            target_lengths: Length of the label sequence (BATCH).

        Returns:
            CTC loss with length penalty.
        """

        # Calculate CTC loss
        ctc_loss = nn.CTCLoss(zero_infinity=True, blank=self.blank, reduction='mean')(
            # log_probs.transpose(1, 0),  # Transpose to (SEQUENCE, BATCH, CLASS)
            log_probs.log_softmax(-1).transpose(1, 0),  # (SEQUENCE, BATCH, CLASS)
            targets, 
            input_lengths, 
            target_lengths
        )

        # Calculate length penalty
        length_penalty = torch.abs(input_lengths.float() - target_lengths.float()) * self.length_penalty_factor

        decoded = ctc_decode(log_probs)
        pred_lengths = torch.tensor([len(x) for x in decoded]).to(device)
        # ##print(decoded, pred_lengths, target_lengths)
        # logprobs_grads = log_probs.log_softmax(-1).argmax(-1).backward()
        length_penalty = torch.abs(pred_lengths.float() - target_lengths.float()) * self.length_penalty_factor
        # ##print(length_penalty)
        # ##print('LOSS:\n{}\nPREDS:\n{}\n, TARGETS:\n{}\n'.format(ctc_loss, log_probs.argmax(-1), targets))
        # Combine CTC loss and length penalty
        total_loss = self.length_penalty_factor* ctc_loss + (1- self.length_penalty_factor)*length_penalty.mean()

        return total_loss
    

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
        
        loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
        loss_list.append(loss)
        pred_txt = ctc_decode(y)
        
        truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
        wer.extend(MyDataset.wer(pred_txt, truth_txt)) 
        cer.extend(MyDataset.cer(pred_txt, truth_txt))              
        if(i_iter % 20 == 0):
            v = 1.0*(time.time()-tic)/(i_iter+1)
            eta = v * (len(val_dataloader)-i_iter) / 3600.0
            
            ##print(''.join(101*'-'))                
            ##print('{:<50}|{:>50}'.format('predict', 'truth'))
            ##print(''.join(101*'-'))                
            # for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                #print('{:<50}|{:>50}'.format(predict, truth))                
            ##print(''.join(101 *'-'))
            ##print('test_iter={},eta={},wer={},cer={}'.format(i_iter,eta,np.array(wer).mean(),np.array(cer).mean()))                
            ##print(''.join(101 *'-'))
            
    # return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())



def train(model, datamodule, optimizer, loss_fn, scheduler=None):
    epochs = 1000
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
            # print(input)
            vid, txt, vid_len, txt_len = input
            vid = vid.to(device)
            txt = txt.to(device)
            vid_len = vid_len.to(device)
            txt_len = txt_len.to(device)

            # vid = input.get('vid').to(device)
            # txt = input.get('txt').to(device)

            # vid_len = input.get('vid_len').to(device)
            # txt_len = input.get('txt_len').to(device)
            batch_size, frames, channels, hight, width = vid.shape
            # ##print(batch_size, frames, channels, hight, width)
            batch_size, seq_length = txt.shape
            vid = vid.permute(0, 2, 1, 3, 4)

            optimizer.zero_grad()
            pred_alignments = model(vid)                                          # [Batch, Seq Length, Class]
            # ##print('OUTPUT SHAPE: ', pred_alignments.shape)
            pred_alignments_for_ctc = pred_alignments.permute(1, 0, -1)               # [Seq Length, Batch, Class]
            txts = [t[t != 0] for t in txt]

            input_length = torch.sum(torch.ones_like(pred_alignments[:, :, 0]), dim=1).int()
            label_length = torch.sum(torch.ones_like(txt), dim=1)
            loss = loss_fn(
                        # pred_alignments.log_softmax(-1).transpose(1, 0), 
                        pred_alignments.transpose(1, 0), 

                        torch.cat(txts), 
                        # txt,
                        # vid_len,
                        input_length, 
                        txt_len)
            
           
            # ##print('STEP LOSS: ', loss)
            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         ##print("Gradients for parameter", name)
            #         ##print(param.grad)
            #     else: ##print('param.grad is None')
            optimizer.step()

            if scheduler is not None:
                scheduler.step(loss)

            tot_iter = it + epoch*len(training_dataloader)
            
            # ##print(pred_alignments)
            pred_txt = ctc_decode(pred_alignments)
            # ##print(pred_txt)
            # ##print(pred_alignments, pred_alignments_for_ctc)
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))
            
            # if(tot_iter % 10 == 0):
            #     v = 1.0*(time.time()-tic)/(tot_iter+1)
            #     eta = (len(training_dataloader)-it)*v/3600.0
                
                # writer.add_scalar('train loss', loss, tot_iter)
                # writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)              
                ##print(''.join(101*'-'))                
                ##print('{:<50}|{:>50}'.format('predict', 'truth'))                
                ##print(''.join(101*'-'))
                
                # for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    ##print('{:<50}|{:>50}'.format(predict, truth))
                ##print(''.join(101*'-'))                
                ##print('epoch={},tot_iter={},eta={},loss={},train_wer={}'.format(epoch, tot_iter, eta, loss, np.array(train_wer).mean()))
                # ##print('epoch={},tot_iter={},eta={},loss={},train_wer={}'.format(epoch, tot_iter, eta, loss, MyDataset.wer(pred_txt, truth_txt).mean()))
                
                ##print(''.join(101*'-'))
                
            # if(tot_iter % 25 == 0):                
            #     (loss, wer, cer) = test(model, datamodule, loss_fn)
            #     ##print('i_iter={},lr={},loss={},wer={},cer={}'
            #         .format(tot_iter,show_lr(optimizer),loss,wer,cer))
                # writer.add_scalar('val loss', loss, tot_iter)                    
                # writer.add_scalar('wer', wer, tot_iter)
                # writer.add_scalar('cer', cer, tot_iter)
                # savename = '{}_loss_{}_wer_{}_cer_{}.pt'.format(opt.save_prefix, loss, wer, cer)
                # (path, name) = os.path.split(savename)
                # if(not os.path.exists(path)): os.makedirs(path)
                # torch.save(model.state_dict(), savename)
                # if(not True):
                #     exit()



        #     tot_iter = it + epoch*len(training_dataloader)

        #     train_loss += loss.item()*batch_size
        # ##print('EPOCH LOSS: ', train_loss/len(training_dataloader.dataset))
        # train_loss_history.append(train_loss/len(training_dataloader.dataset))




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

    model = Model(len(MyDataset.characters)+1)
    ##print(model)
    optimizer = Adam(params=model.parameters(), 
                 lr=1e-8,
                 amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)

    ##print('LEN LETTERS:', len(MyDataset.characters))
    # blank=len(MyDataset.characters),
    loss_fn = nn.CTCLoss(zero_infinity=True, reduction='mean')
    # loss_fn = CTCLossWithLengthPenalty(length_penalty_factor=0.5)
    ##print(optimizer)
    train(model, datamodule, optimizer, loss_fn, scheduler)
