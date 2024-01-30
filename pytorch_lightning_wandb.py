import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

# wandb 초기화
wandb.init(project="MNIST_with_Pytorch-lightning")

# 데이터 전처리 설정
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# 데이터셋을 5개로 분할
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# MNIST 데이터셋 불러오기
full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# 첫 번째 fold를 검증 데이터셋으로 사용
train_ids, valid_ids = next(iter(kfold.split(full_train_dataset)))

train_dataset = Subset(full_train_dataset, train_ids)
val_dataset = Subset(full_train_dataset, valid_ids)

# 데이터로더 설정
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 모델 정의
class Net(pl.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        self.log('val_loss', loss)
        preds = output.argmax(dim=1)
        acc = (preds == target).float().mean()
        self.log('val_acc', acc)
        return acc

    def test_step(self, batch):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        self.log('test_loss', loss)
        preds = output.argmax(dim=1)
        acc = (preds == target).float().mean()
        self.log('test_acc', acc)
        return acc

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-5, weight_decay=5e-4, eps=5e-9)

# wandb 로거 설정       
wandb_logger = WandbLogger()

model = Net()

trainer = pl.Trainer(accelerator="auto", max_epochs=50, logger=wandb_logger)
trainer.fit(model, train_loader, val_loader)
trainer.test(dataloaders=test_loader)

wandb.finish()