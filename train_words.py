import argparse

import psutil
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# from src.checkpoint import load_checkpoint
from src.model.lrw_model import E2E
import wandb
import gc
# wandb.login()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/datasets/lrw")
    parser.add_argument("--checkpoint_dir", type=str, default='data/checkpoints/lrw')
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--words", type=int, default=10)
    # parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    # parser.add_argument("--resnet", type=int, default=18)
    # parser.add_argument("--pretrained", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--use_amp", default=False, action='store_true')
    args = parser.parse_args()
    # print('HERE: ', args)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        # save_best_only=True,
        save_top_k=5,
        monitor='val_acc',
        mode='max',
        save_last=True,
        filename="{epoch}",
    )

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=10,
        mode='max',
    )


    args.workers = psutil.cpu_count(logical=False) if args.workers == None else args.workers
    # args.pretrained = False if args.checkpoint != None else args.pretrained
    model = E2E(
        "/home/sadevans/space/LRModel/config_ef.yaml",
        hparams=args,
        in_channels=1,
        augmentations=False,
        # query=query,
    )
    # logger = WandbLogger(
    #     project=f'lipreading_lrw_classification_{args.words}words',
    #     model=model,
    # )

    logger = WandbLogger(name="exp", project=f'lipreading_lrw_classification_{args.words}words',save_dir=f"/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints/exp1")
    logger.watch(model = model, log='gradients',log_graph=True)

    # model.logger = logger
    seed_everything(args.seed)
    callbacks = [checkpoint_callback, early_stop_callback]
    trainer = Trainer(
        # seed=args.seed,
        logger=logger,
        gpus=-1,
        max_epochs=args.epochs,
        callbacks=callbacks,
        log_every_n_steps=5,
        # early_stop_callback=early_stop_callback,
        # checkpoint_callback=checkpoint_callback,
        # use_amp=args.use_amp,
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters: {trainable_params}")
    logger.log_hyperparams({"trainable_params": trainable_params})
    logger.log_hyperparams(args)

    if args.checkpoint != None:
        logs = trainer.validate(model, checkpoint=args.checkpoint)
        logger.log_metrics({'val_acc': logs['val_acc'], 'val_loss': logs['val_loss']})
        print(f"Initial val_acc: {logs['val_acc']:.4f}")

    trainer.fit(model)
    # logger.save_file(checkpoint_callback.last_checkpoint_path)

    # python train_words.py --data "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/lipread_mp4" --checkpoint_dir "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints" --lr 1e-6