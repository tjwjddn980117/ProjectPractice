import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ..utils.config import CFG
from ..datasets.Datasets import CustomDataset
from ..datasets.DatasetsFn import train_collate_fn, val_collate_fn
from ..models.pytorch_model import CustomModel

def train(epoch, total_epoch, train_dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}/{total_epoch}")
    for i, (data, target) in progress_bar:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

         # 손실 및 정확도 계산
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        # tqdm의 설명 부분을 업데이트
        progress_bar.set_postfix(loss=running_loss/(i+1), accuracy=100.*correct/total)
    # 에포크 완료 후 출력
    print(f"Epoch {epoch}, Loss: {running_loss/len(train_dataloader):.4f}, Accuracy: {correct/total:.4f}")
            
# 검증 함수
def valid(val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    val_loss /= len(val_loader.dataset)
    print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    wandb.log({"Valid Accuracy": 100. * correct / len(val_loader.dataset), "Valid Loss": val_loss})
            

# 테스트 함수
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    wandb.log({"Test Accuracy": 100. * correct / len(test_loader.dataset), "Test Loss": test_loss})

# 데이터셋을 5개로 분할
skf = StratifiedKFold(n_splits=CFG.N_SPLIT, random_state=CFG.SEED, shuffle=True)

# MNIST 데이터셋 불러오기
#full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
#test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# 데이터와 레이블 분리
#train_data = [data for data, _ in full_train_dataset]
#train_labels = [label for _, label in full_train_dataset]

train_df = pd.read_csv('./train.csv')
le = LabelEncoder()
# just mapping the str to int index. 
train_df['class'] = le.fit_transform(train_df['label'])


for fold_idx, (train_index, val_index) in enumerate(skf.split(train_df, train_df['class'])):
    train_fold_df = train_df.loc[train_index,:]
    val_fold_df = train_df.loc[val_index,:]

    train_dataset = CustomDataset(train_fold_df, 'img_path', mode='train')
    val_dataset = CustomDataset(val_fold_df, 'img_path', mode='val')

    train_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=CFG.BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, collate_fn=val_collate_fn, batch_size=CFG.BATCH_SIZE*2)

    # 모델, 손실함수, 최적화함수 설정
    model = CustomModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-5, weight_decay=5e-4, eps=5e-9)
    
    # wandb에 모델, 최적화 함수 로그
    wandb.watch(model, log="all")
    wandb.config.update({"Optimizer": "ADAM", "Learning Rate": 0.01, "Momentum": 0.5})

    for epoch in range(CFG.EPOCHS):
        train(epoch, CFG.EPOCHS, train_dataloader, criterion, optimizer)
        valid(val_dataloader)

#test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)
#test()