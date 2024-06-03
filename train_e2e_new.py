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
from scheduler import WarmupCosineScheduler
import os


def get_classes(path):
    words = os.listdir(path)
    words_classes= {}
    for i, word in enumerate(words):
        word = word.lower()
        words_classes[word] = i

    return words_classes

def stochastic_depth(model, x, survival_prob):
    if not model.training:
        return x
    binary_tensor = torch.rand(x.shape(0), 1, 1, 1, device=x.device) < survival_prob
    x = x / survival_prob
    x = x * binary_tensor
    return x


def adjust_learning_rate(optimizer, epoch, step, total_steps):
    if epoch < 3:
        lr = 1e-6 + (0.18 - 1e-6) * (step + epoch * total_steps) / (3 * total_steps)
    else:
        lr = 0.18 * (0.97 ** ((epoch - 3 + step / total_steps) / 2.4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch, model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    total_steps = len(dataloader)
    # for inputs, labels in dataloader:
    for step, batch in enumerate(dataloader):
        # adjust_learning_rate(optimizer, epoch, step, total_steps)
        inputs = batch[0].to(device)
        labels = batch[1]

        # inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs, classification=True)
        # print(outputs.shape)
        # outputs = stochastic_depth(model, outputs, 0.8)
        # print(outputs)
        labels_num = [words_classes[word] for word in labels]
        
        targets_tensor = torch.LongTensor(labels_num)
        # print(outputs.shape)
        loss = criterion(outputs, targets_tensor.to(device))
        
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # ema.update_parameters(model)
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    
    return epoch_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            labels = batch[1]
            
            outputs = model(inputs, classification=True)

            labels_num = [words_classes[word] for word in labels]
        
            targets_tensor = torch.LongTensor(labels_num)
            # print("Index of max element, true element: ", torch.argmax(outputs, dim=1), targets_tensor)

            loss = criterion(outputs, targets_tensor.to(device))
            
            running_loss += loss.item() * inputs.size(0)
        
    # print(outputs.shape, outputs.detach()[0])
    print("Index of max element, true element: ", torch.argmax(outputs.detach()[0]), targets_tensor[0])
    
    epoch_loss = running_loss / len(dataloader.dataset)
    
    return epoch_loss


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, num_epochs):
    for epoch in range(num_epochs):
        train_loss = train(epoch, model, train_dataloader, criterion, optimizer, scheduler, device)
        val_loss = validate(model, val_dataloader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
    return model


if __name__ == "__main__":

    datamodule = DataModule(
        "video", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/for_model",
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/for_model/labels/lrw_train_transcript_lengths_seg1.6s.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/for_model/labels/lrw_val_transcript_lengths_seg1.6s.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/for_model/labels/lrw_val_transcript_lengths_seg1.6s.csv"
    )

    model = E2E("/home/sadevans/space/LRModel/config_ef.yaml", num_classes=500, efficient_net_size="T")

    model = model.to(device='cuda' if torch.cuda.is_available() else 'cpu')
   
    ema = torch.optim.swa_utils.AveragedModel(model, avg_fn=lambda avg, new, decay: 0.9999*avg + (1-0.9999)*new)

    # optimizer = torch.optim.AdamW([{"name": "model", "params": model.parameters(), "lr": 1e-6}], weight_decay=0.003, betas=(0.9, 0.98))
    optimizer = RMSprop([{"name": "model", "params": model.parameters(), "lr": 1e-6}], weight_decay=1e-5, alpha=0.9, momentum= 0.9)
    
    scheduler = WarmupCosineScheduler(optimizer, 3, 75, len(datamodule.train_dataloader()))
    
    loss_fn = nn.CrossEntropyLoss().to(device)

    words_classes = get_classes("/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/lipread_mp4")

    model = train_model(model, datamodule.train_dataloader(), datamodule.val_dataloader(), loss_fn, optimizer, scheduler, device, 75)
