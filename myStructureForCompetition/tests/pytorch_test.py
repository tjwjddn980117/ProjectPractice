import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import wandb

from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from glob import glob
from ..utils.config import CFG
from ..datasets.Datasets import CustomDataset
from ..datasets.DatasetsFn import test_collate_fn

from ..models.pytorch_model import CustomModel
from ..trainers.pytorch_trainer import criterion, le

# 테스트 함수
def test(model, test_dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    progress_bar = tqdm(test_dataloader, total=len(test_dataloader))
    with torch.no_grad():
        for data, target in progress_bar:
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            preds = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += preds.eq(target.data.view_as(preds)).cpu().sum()

    test_loss /= len(test_dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))
    wandb.log({"Test Accuracy": 100. * correct / len(test_dataloader.dataset), "Test Loss": test_loss})

    return preds

test_df = pd.read_csv('./data/test.csv')
test_df['img_path'] = test_df['img_path'].apply(lambda x: os.path.join('./data', x))

if not len(test_df) == len(os.listdir('./data/test')):
    raise ValueError()

test_dataset = CustomDataset(test_df, 'img_path', mode='inference')
test_dataloader = DataLoader(test_dataset, collate_fn=test_collate_fn, batch_size=CFG.BATCH_SIZE*2)

wandb.init(project="Sample_Practice")

fold_preds = []
for checkpoint_path in glob('./checkpoints/swinv2-large-resize*.ckpt'):
    model = CustomModel()
    # wandb 초기화
    wandb.watch(model, log="all")
    model.load_state_dict(torch.load(checkpoint_path))
    preds = test(model, test_dataloader)
    preds = torch.cat(preds,dim=0).detach().cpu().numpy().argmax(1)
    fold_preds.append(preds)
pred_ensemble = list(map(lambda x: np.bincount(x).argmax(),np.stack(fold_preds,axis=1)))

submission = pd.read_csv('./data/sample_submission.csv')

submission['label'] = le.inverse_transform(pred_ensemble)

submission.to_csv('./submissions/swinv2_large_resize.csv',index=False)