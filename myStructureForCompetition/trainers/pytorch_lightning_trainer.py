import gc
import pandas as pd
import torch
import pytorch_lightning as L

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import timm

from ..utils.config import CFG
from ..datasets.Datasets import CustomDataset
from ..datasets.DatasetsFn import train_collate_fn, val_collate_fn
from ..models.pytorch_lightning_model import LitCustomModel

train_df = pd.read_csv('./train.csv')
le = LabelEncoder()
# just mapping the str to int index. 
train_df['class'] = le.fit_transform(train_df['label'])

skf = StratifiedKFold(n_splits=CFG.N_SPLIT, random_state=CFG.SEED, shuffle=True)

for fold_idx, (train_index, val_index) in enumerate(skf.split(train_df, train_df['class'])):
        train_fold_df = train_df.loc[train_index,:]
        val_fold_df = train_df.loc[val_index,:]

        train_dataset = CustomDataset(train_fold_df, 'img_path', mode='train')
        val_dataset = CustomDataset(val_fold_df, 'img_path', mode='val')

        train_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=CFG.BATCH_SIZE)
        val_dataloader = DataLoader(val_dataset, collate_fn=val_collate_fn, batch_size=CFG.BATCH_SIZE*2)

        model = timm.create_model('eva_large_patch14_196.in22k_ft_in22k_in1k', pretrained=True)
        lit_model = LitCustomModel(model)

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
        trainer = L.Trainer(max_epochs=100, accelerator='auto', precision=32, callbacks=[checkpoint_callback, earlystopping_callback], val_check_interval=0.5)
        trainer.fit(lit_model, train_dataloader, val_dataloader)

        model.cpu()
        lit_model.cpu()
        del model, lit_model, checkpoint_callback, earlystopping_callback, trainer
        gc.collect()
        torch.cuda.empty_cache()