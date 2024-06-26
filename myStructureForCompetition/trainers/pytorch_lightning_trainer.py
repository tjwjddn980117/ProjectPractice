import gc
import pandas as pd
import  os

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold

import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import timm

from utils.config import CFG
from utils.init_sharing_resources import load_label_encoder, load_wandb_id
from utils.functions import wrap_loader_with_tqdm
from datasets.Datasets import CustomDataset
from datasets.DatasetsFn import train_collate_fn, val_collate_fn
from models.pytorch_lightning_model import LitCustomModel

def trainer():
    train_df = pd.read_csv('./train.csv')
    #le = LabelEncoder()
    le = load_label_encoder(CFG.LABEL_ENCODER_NAME)
    # just mapping the str to int index. 
    train_df['class'] = le.fit_transform(train_df['label'])

    skf = StratifiedKFold(n_splits=CFG.N_SPLIT, random_state=CFG.SEED, shuffle=True)

    # K-fold (StratifiedKFold) method. 
    for fold_idx, (train_index, val_index) in enumerate(skf.split(train_df, train_df['class'])):
            train_fold_df = train_df.loc[train_index,:]
            val_fold_df = train_df.loc[val_index,:]

            train_dataset = CustomDataset(train_fold_df, 'img_path', mode='train')
            val_dataset = CustomDataset(val_fold_df, 'img_path', mode='val')

            train_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=CFG.BATCH_SIZE)
            val_dataloader = DataLoader(val_dataset, collate_fn=val_collate_fn, batch_size=CFG.BATCH_SIZE*2)

            train_dataloader = wrap_loader_with_tqdm(train_dataloader)
            val_dataloader = wrap_loader_with_tqdm(val_dataloader)

            wandb_logger = WandbLogger()

            # you should modify this part. 
            model = timm.create_model('eva_large_patch14_196.in22k_ft_in22k_in1k', pretrained=True)
            lit_model = LitCustomModel(model)

            checkpoint_dir = './checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_callback = ModelCheckpoint(
                monitor='val_score',
                mode='max',
                dirpath='./checkpoints/',
                filename=f'eva-large-196-resize-fold_idx={fold_idx}'+'-{epoch:02d}-{train_loss:.4f}-{val_loss:.4f}-{val_score:.4f}',
                save_top_k=1,
                save_weights_only=True,
                verbose=True
            )

            earlystopping_callback = EarlyStopping(monitor="val_score", mode="max", patience=6)
            trainer = L.Trainer(max_epochs=100, accelerator='auto', precision=32, callbacks=[checkpoint_callback, earlystopping_callback], val_check_interval=0.5, logger=wandb_logger)
            trainer.fit(lit_model, train_dataloader, val_dataloader)

            model.cpu()
            lit_model.cpu()
            del model, lit_model, checkpoint_callback, earlystopping_callback, trainer
            gc.collect()
            torch.cuda.empty_cache()