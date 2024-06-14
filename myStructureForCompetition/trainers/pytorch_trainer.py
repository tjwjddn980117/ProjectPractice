import pandas as pd
import os
from tqdm import tqdm
import wandb

from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils.config import CFG
from utils.init_sharing_resources import load_label_encoder, load_wandb_id
from datasets.Datasets import CustomDataset
from datasets.DatasetsFn import train_collate_fn, val_collate_fn
from models.pytorch_model import CustomModel
from models.pytorch_model_torchvision_swin_transformer import Swin_B32, print_Swin_B32
import timm


def train(model, epoch, total_epoch, train_dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}/{total_epoch}")
    for i, (data, target) in progress_bar:
        data = data.to(CFG.DEVICE)
        target = target.to(CFG.DEVICE)
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
        # wandb에 손실 및 정확도 로깅
        # wandb.log({"Train Loss": running_loss/(i+1), "Train Accuracy": 100.*correct/total})
    # 에포크 완료 후 출력
    print(f"Epoch {epoch}, Loss: {running_loss/len(train_dataloader):.4f}, Accuracy: {correct/total:.4f}")
    wandb.log({"Epoch": epoch, "Loss": running_loss/len(train_dataloader), "Accuracy": correct/total})
            
# 검증 함수
def valid(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(CFG.DEVICE)
            target = target.to(CFG.DEVICE)
            output = model(data)
            val_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    val_loss /= len(val_loader.dataset)
    print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    wandb.log({"Valid Accuracy": 100. * correct / len(val_loader.dataset), "Valid Loss": val_loss})

    return val_loss

def trainer(load_file=False):
    # 데이터셋을 5개로 분할
    skf = StratifiedKFold(n_splits=CFG.N_SPLIT, random_state=CFG.SEED, shuffle=True)

    csv_path = CFG.PROJECT_PATH + 'data\\train.csv'
    train_df = pd.read_csv(csv_path)
    # le = LabelEncoder()
    le = load_label_encoder(CFG.LABEL_ENCODER_NAME)
    # just mapping the str to int index. 
    train_df['class'] = le.fit_transform(train_df['label'])

    load_wandb_id(CFG.WANDB_ID_NAME)

    # K-fold (StratifiedKFold) method. 
    for fold_idx, (train_index, val_index) in enumerate(skf.split(train_df, train_df['class'])):
        train_fold_df = train_df.loc[train_index,:]
        val_fold_df = train_df.loc[val_index,:]

        train_dataset = CustomDataset(train_fold_df, 'img_path', mode='train')
        val_dataset = CustomDataset(val_fold_df, 'img_path', mode='val')

        train_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, shuffle=True, batch_size=CFG.BATCH_SIZE)
        val_dataloader = DataLoader(val_dataset, collate_fn=val_collate_fn, batch_size=CFG.BATCH_SIZE*2)

        # 모델, 손실함수, 최적화함수 설정
        model = Swin_B32().to(CFG.DEVICE)
        print_Swin_B32(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=1e-5, weight_decay=5e-4, eps=5e-9)

        # 더미 배치를 사용하여 파라미터 초기화
        dummy_input = torch.randn(CFG.BATCH_SIZE, CFG.CHANNELS, CFG.WIDTH, CFG.HEIGHT).to(device=CFG.DEVICE)  # 입력 텐서 크기 조정 필요
        model(dummy_input)  # forward 호출로 파라미터 초기화

        if load_file:
            model.load_state_dict(torch.load(''))
        # wandb에 모델, 최적화 함수 로그
        wandb.watch(model, log="all")
        wandb.config.update({"Optimizer": "ADAM", "Learning Rate": 0.01, "Momentum": 0.5})

        best_val_loss = float('inf')
        for epoch in range(CFG.EPOCHS):
            train(model, epoch, CFG.EPOCHS, train_dataloader, criterion, optimizer)
            val_loss = valid(model, val_dataloader, criterion)

            checkpoint_dir = './checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 모델 가중치 저장
            checkpoint_path = os.path.join(checkpoint_dir, f'fold{fold_idx}_epoch{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_path)

            # Validation 성능이 향상될 때마다 가장 좋은 모델 가중치 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = os.path.join(checkpoint_dir, 'best_epoch.pt')
                torch.save(model.state_dict(), best_checkpoint_path)

    #test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)
    #test()