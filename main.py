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
from lipnet import MyModel
import torch.optim as optim
import re
import json
# from tensorboardX import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function


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
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]


def train(model):
    print("in train")
    datamodule = DataModule(
        "video", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_",
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_train_transcript_lengths_seg24s_0to100_5000units.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_val_transcript_lengths_seg24s_0to100_5000units.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_val_transcript_lengths_seg24s_0to100_5000units.csv"
    )
    max_epochs = 2
    loader = datamodule.train_dataloader()
    print("loader: ", loader)
    optimizer = optim.Adam(model.parameters(),
                lr = 1e-3,
                weight_decay = 0.,
                amsgrad = True)
                
    # print('num_train_data:{}'.format(len(dataset.data)))    
    crit = nn.CTCLoss(reduction="sum", zero_infinity=True)
    tic = time.time()
    
    train_wer = []
    for epoch in range(max_epochs):
        # print("in train loop")
        model.train()
        for (i_iter, input) in enumerate(loader):
            # print(epoch)
            # model.train()
            

            vid = input.get('vid').to(device)
            batch_size, frames, channels, hight, width = vid.shape
            print(batch_size, frames, channels, hight, width)
        #     vid = vid.permute(0, 2, 1, 3, 4)
        #     # break
        #     print("vid shape: ", vid.shape)
        #     txt = input.get('txt').to(device)
        #     print("txt shape: ", txt.shape)
        #     vid_len = input.get('vid_len').to(device)
        #     txt_len = input.get('txt_len').to(device)
        #     # print('here: ', vid[58], txt[58], MyDataset.arr2txt(txt[58], start=1))
        #     # break
        #     optimizer.zero_grad()
        #     # print(vid)
           
        #     # y = net(vid)
        #     y = model(vid)

        #     # print(model.weights)
        #     if torch.any(torch.isnan(y)):
        #         print('THERE ARE NANS AFTER')
        #     # print(y)
        #     # print('Y SHAPE: ', y.shape)
        #     # print('TXT SHAPE: ', txt.squeeze().shape)
        #     # print(txt)
        #     # print(y.transpose(0, 1).shape)
        #     # print(y.transpose(0, 1).log_softmax(-1))
        #     # logits = y.transpose(0, 1).log_softmax(-1)
        #     # sequence, batch_size, channels,  = logits.size()
        #     # print(batch_size, channels, sequence)
        #     # print(vid_len.view(-1).shape)
        #     txts = [t[t != -1] for t in txt]
        #     # l = 0
        #     # for t in txts:
        #     #     l+= t.shape[0]
        #     # print(l)
        #     # print(type(txts), txts)
        #     # for t in txt:
        #     #     print(t)
        #     #     print('here: ', type(t[t!=-1]))
        #     # print(txt[0][0])
        #     # print(type(txt[0][0]))
        #     # txts = [t[t != -1] for t in txt]
        #     # break
        #     txts_pad = torch.cat(txts)
        #     print(txts_pad.shape)

        #     # olens = to_device(y, torch.LongTensor([len(s) for s in txts]))
        #     # print(olens.shape, txt_len.view(-1).shape)
        #     # break
        #     # loss = crit(y.transpose(0, 1).log_softmax(-1), txt.squeeze(), vid_len.view(-1), txt_len.view(-1))
        #     # loss = crit(y.transpose(0, 1).log_softmax(-1), txts, vid_len.view(-1), olens)
        #     # loss = crit(y.transpose(0, 1).log_softmax(-1), txts_pad, vid_len.view(-1), txt_len.view(-1))
        #     print(y.transpose(0, 1).log_softmax(-1).shape, txts_pad.shape, vid_len.shape, txt_len.shape)
        #     loss = custom_ctc_loss(y.transpose(0, 1).log_softmax(-1), txts_pad, vid_len.view(-1), txt_len.view(-1), zero_infinity=True)
        #     # print(loss)

        #     loss = loss / y.size(1)

        #     print('LOSS: ', loss)
        #     # break
        #     # loss.retain_grad()
        #     loss.backward()
        #     # print('LOSS GRADS: ', loss.grad.data)
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             print("Gradients for parameter", name)
        #             print(param.grad)
        #         else: print('param.grad is None')

        #     # if(True):
        #     optimizer.step()
            
        #     tot_iter = i_iter + epoch*len(loader)
            
        #     pred_txt = ctc_decode(y)
        #     print('PRED TXT: ', pred_txt)
            
        #     truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
        #     print('TRUTH TEXT: ', truth_txt)
        #     train_wer.extend(MyDataset.wer(pred_txt, truth_txt))
        #     print('TRAIN WER: ', MyDataset.wer(pred_txt, truth_txt))
        #     # break
        # # break
        #     if(tot_iter % 10 == 0):
        #         v = 1.0*(time.time()-tic)/(tot_iter+1)
        #         eta = (len(loader)-i_iter)*v/3600.0
                
        #         # writer.add_scalar('train loss', loss, tot_iter)
        #         # writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)              
        #         print(''.join(101*'-'))                
        #         print('{:<50}|{:>50}'.format('predict', 'truth'))                
        #         print(''.join(101*'-'))
                
        #         for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
        #             print('{:<50}|{:>50}'.format(predict, truth))
        #         print(''.join(101*'-'))                
        #         print('epoch={},tot_iter={},eta={},loss={},train_wer={}'.format(epoch, tot_iter, eta, loss, np.array(train_wer).mean()))
        #         print(''.join(101*'-'))
                
            # if(tot_iter % opt.test_step == 0):                
            #     (loss, wer, cer) = test(model, net)
            #     print('i_iter={},lr={},loss={},wer={},cer={}'
            #         .format(tot_iter,show_lr(optimizer),loss,wer,cer))
            #     writer.add_scalar('val loss', loss, tot_iter)                    
            #     writer.add_scalar('wer', wer, tot_iter)
            #     writer.add_scalar('cer', cer, tot_iter)
            #     savename = '{}_loss_{}_wer_{}_cer_{}.pt'.format(opt.save_prefix, loss, wer, cer)
            #     (path, name) = os.path.split(savename)
            #     if(not os.path.exists(path)): os.makedirs(path)
            #     torch.save(model.state_dict(), savename)
            #     if(not opt.is_optimize):
            #         exit()




if __name__ == "__main__":
    model = MyModel()
    model = model.to(device)
    # net = nn.DataParallel(model).to(device)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    train(model)