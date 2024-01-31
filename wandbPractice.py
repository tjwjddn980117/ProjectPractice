# 필요한 라이브러리들을 불러옵니다.
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# wandb 초기화
wandb.init(project="MNIST_Practice_with_pictures")

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
class Net(nn.Module):
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

# 모델, 손실함수, 최적화함수 설정
model = Net()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=1e-5, weight_decay=5e-4, eps=5e-9)

# wandb에 모델, 최적화 함수 로그
wandb.watch(model, log="all")
wandb.config.update({"Optimizer": "ADAM", "Learning Rate": 0.01, "Momentum": 0.5})

# 학습 함수
def train(epoch):
    model.train()
    #temp=0
    for batch_idx, (data, target) in enumerate(train_loader):
        #temp+=1
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 150 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    #print(temp)
# 검증 함수
def valid():
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            # 예측 이미지와 라벨을 WandB에 로깅
            pred_image = data[0].cpu().numpy()
            pred_label = pred[0].cpu().numpy()
            true_label = target[0].cpu().numpy()
            wandb.log({
                "Predicted Images": [wandb.Image(pred_image, caption='Pred: {}'.format(pred_label))],
                "True Images": [wandb.Image(pred_image, caption='True: {}'.format(true_label))]
            })

    val_loss /= len(val_loader.dataset)
    print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
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
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            # 예측 이미지와 라벨을 WandB에 로깅
            pred_image = data[0].cpu().numpy()
            pred_label = pred[0].cpu().numpy()
            true_label = target[0].cpu().numpy()
            wandb.log({
                "Predicted Images": [wandb.Image(pred_image, caption='Pred: {}'.format(pred_label))],
                "True Images": [wandb.Image(pred_image, caption='True: {}'.format(true_label))]
            })

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    wandb.log({"Test Accuracy": 100. * correct / len(test_loader.dataset), "Test Loss": test_loss})

# 모델 학습 및 테스트
for epoch in range(1, 50 + 1):
    train(epoch)
    valid()
test()
