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
from lipnet import LipNet
import torch.optim as optim
import re
import json
# from tensorboardX import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()  

def ctc_decode(y):
    result = []
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]


def train(model, net):
    print("in train")
    datamodule = DataModule(
        "video", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_",
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_train_transcript_lengths_seg24s_0to100_5000units.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_val_transcript_lengths_seg24s_0to100_5000units.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_val_transcript_lengths_seg24s_0to100_5000units.csv"
    )

    loader = datamodule.train_dataloader()
    print("loader: ", loader)
    optimizer = optim.Adam(model.parameters(),
                lr = 2e-5,
                weight_decay = 0.,
                amsgrad = True)
                
    # print('num_train_data:{}'.format(len(dataset.data)))    
    crit = nn.CTCLoss()
    tic = time.time()
    
    train_wer = []
    for epoch in range(2):
        print("in train loop")
        for (i_iter, input) in enumerate(loader):
            model.train()
            vid = input.get('vid').to(device)
            vid = vid.permute(0, 2, 1, 3, 4)

            print("vid shape: ", vid.shape)
            txt = input.get('txt').to(device)
            print("txt shape: ", txt.shape)
            vid_len = input.get('vid_len').to(device)
            txt_len = input.get('txt_len').to(device)
            
            optimizer.zero_grad()
            y = net(vid)
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
            loss.backward()
            if(True):
                optimizer.step()
            
            tot_iter = i_iter + epoch*len(loader)
            
            pred_txt = ctc_decode(y)
            
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))
            
            if(tot_iter % 10 == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(loader)-i_iter)*v/3600.0
                
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
    model = LipNet()
    model = model.to(device)
    net = nn.DataParallel(model).to(device)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    train(model, net)