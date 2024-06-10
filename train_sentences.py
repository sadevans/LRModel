import argparse

import psutil
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
# from lightning_datamodule import DataModule
from datamodule import DataModule
# from src.checkpoint import load_checkpoint
from src.model.sent_model import E2E
import wandb
import gc
# wandb.login()
# from ray import tune
# from ray import train
# from ray.train import Checkpoint, get_checkpoint
# from ray.tune.schedulers import ASHAScheduler, FIFOScheduler
# import ray.cloudpickle as pickle
import os
from functools import partial
from datetime import datetime

def init_train(config, args=None):
    print(config['batch_size'])
    args.batch_size = int(config['batch_size'])
    args.lr = float(config['lr'])
    # args.dropout = float(config['dropout'])
    modelmodule = E2E(
        "/home/sadevans/space/LRModel/config_ef.yaml",
        hparams=args,
        dropout=float(config['dropout']),
        in_channels=1,
        augmentations=False,
    )

    datamodule = DataModule(
        "video", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_",
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_train_transcript_lengths_seg24s_0to100_5000units.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_val_transcript_lengths_seg24s_0to100_5000units.csv", 
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_val_transcript_lengths_seg24s_0to100_5000units.csv"
    )

    name = f"exp_lr{args.lr}_batch_size{args.batch_size}_dropout{config['dropout']}"
    os.makedirs(f"/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints/{name}", exist_ok=True)
    # logger = WandbLogger(name=name, \
    #                      project=f'lipreading_sentences',\
    #                         save_dir=f"/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints/{name}")
    # logger.watch(model = modelmodule, log='gradients',log_graph=True)

    seed_everything(args.seed)
    callbacks = [checkpoint_callback, early_stop_callback]
    trainer = Trainer(
        # seed=args.seed,
        # logger=logger,
        gpus=-1,
        max_epochs=args.epochs,
        callbacks=callbacks,
        log_every_n_steps=5,
        # early_stop_callback=early_stop_callback,
        # checkpoint_callback=checkpoint_callback,
        # use_amp=args.use_amp,
    )
    trainable_params = sum(p.numel() for p in modelmodule.parameters() if p.requires_grad)

    print(f"Trainable parameters: {trainable_params}")
    # logger.log_hyperparams({"trainable_params": trainable_params})
    # logger.log_hyperparams(args)

    if args.checkpoint != None:
        # logs = trainer.validate(modelmodule, checkpoint=args.checkpoint)
        # logger.log_metrics({'val_acc': logs['val_acc'], 'val_loss': logs['val_loss']})
        print(f"Initial val_acc: {logs['val_acc']:.4f}")

    # trainer.fit(modelmodule, datamodule)
    trainer.fit(model=modelmodule, datamodule=datamodule)





if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/datasets/lrw")
    parser.add_argument("--checkpoint_dir", type=str, default='data/checkpoints/lrw')
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=11)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--words", type=int, default=10)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    args.modality = "video"
    args.label_dir = 'labels'
    args.root_dir = "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_"
    args.train_file = "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_train_transcript_lengths_seg24s_0to100_5000units.csv"
    args.val_file = "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_valid_transcript_lengths_seg24s_0to100_5000units.csv"
    args.test_file = "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_valid_transcript_lengths_seg24s_0to100_5000units.csv"

    name = f"exp_lr{args.lr}_batch_size{args.batch_size}_dropout{args.dropout}"

    ttime = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    save_checkpoint_dir = os.makedirs(args.checkpoint_dir + '/' + ttime + '/' + name, exist_ok=True) if args.checkpoint_dir else None
    checkpoint_callback = ModelCheckpoint(
        monitor="val_wer",
        mode="min",
        dirpath=save_checkpoint_dir,
        save_last=True,
        filename="{epoch}",
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        mode='min',
    )


    args.workers = psutil.cpu_count(logical=False) if args.workers == None else args.workers

    modelmodule = model = E2E(
        "/home/sadevans/space/LRModel/config_ef.yaml",
        hparams=args,
        dropout=args.dropout,
        in_channels=1,
        augmentations=False,
    )

    logger = WandbLogger(name=name, \
                         project=f'lipreading_sentences',\
                            save_dir=f"/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints/{name}")
    logger.watch(model = model, log='gradients',log_graph=True)

    seed_everything(args.seed)
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]
    trainer = Trainer(
        logger=logger,
        gpus=-1,
        max_epochs=args.epochs,
        callbacks=callbacks,
        log_every_n_steps=5,
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters: {trainable_params}")
    logger.log_hyperparams({"trainable_params": trainable_params})
    logger.log_hyperparams(args)

    if args.checkpoint != None:
        logs = trainer.validate(model, checkpoint=args.checkpoint)
        logger.log_metrics({'val_wer': logs['val_wer'], 'val_loss': logs['val_loss']})
        print(f"Initial val_wer: {logs['val_wer']:.4f}")

    # trainer.fit(modelmodule, datamodule)
    # modelmodule = modelmodule.load_from_checkpoint("/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints/epoch=9-v4.ckpt")
    # trainer.fit(modelmodule)
    ckpt = torch.load("/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints/exp_lr0.001_batch_size16_dropout0.2/lipreading_sentences/k5mz333h/checkpoints/epoch=3.ckpt", map_location=lambda storage, loc: storage)["state_dict"]
    # print(ckpt.keys())
    modelmodule.load_state_dict(ckpt)
    # modelmodule.load_state_dict(torch.load("/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints/epoch=9-v4.ckpt", map_location=lambda storage, loc: storage))
    trainer.test(modelmodule)

    # logger.save_file(checkpoint_callback.last_checkpoint_path)

    # python train_words.py --data "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/lipread_mp4" --checkpoint_dir "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints" --lr 1e-6