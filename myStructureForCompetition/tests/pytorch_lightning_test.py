import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import torch
import pytorch_lightning as L

from glob import glob
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import Swinv2Config, Swinv2Model, AutoImageProcessor, AutoModelForImageClassification

from utils.config import CFG
from utils.functions import wrap_loader_with_tqdm
from datasets.Datasets import CustomDataset
from datasets.DatasetsFn import test_collate_fn
from models.pytorch_lightning_model import LitCustomModel
from trainers.pytorch_lightning_trainer import le

test_df = pd.read_csv('./data/test.csv')
test_df['img_path'] = test_df['img_path'].apply(lambda x: os.path.join('./data', x))

if not len(test_df) == len(os.listdir('./data/test')):
    raise ValueError()

test_dataset = CustomDataset(test_df, 'img_path', mode='inference')
test_dataloader = DataLoader(test_dataset, collate_fn=test_collate_fn, batch_size=CFG.BATCH_SIZE*2)
test_dataloader = wrap_loader_with_tqdm(test_dataloader)

fold_preds = []
for checkpoint_path in glob('./checkpoints/swinv2-large-resize*.ckpt'):
    model = Swinv2Model.from_pretrained("microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft")
    lit_model = LitCustomModel.load_from_checkpoint(checkpoint_path, model=model)
    trainer = L.Trainer( accelerator='auto', precision=32)
    preds = trainer.predict(lit_model, test_dataloader)
    preds = torch.cat(preds,dim=0).detach().cpu().numpy().argmax(1)
    fold_preds.append(preds)
pred_ensemble = list(map(lambda x: np.bincount(x).argmax(),np.stack(fold_preds,axis=1)))

submission = pd.read_csv('./data/sample_submission.csv')

submission['label'] = le.inverse_transform(pred_ensemble)

submission.to_csv('./submissions/swinv2_large_resize.csv',index=False)