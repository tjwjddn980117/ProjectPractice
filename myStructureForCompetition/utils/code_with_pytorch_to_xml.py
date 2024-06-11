from torch import nn
import torch
import openvino as ov
#hyp parameters
dataset_path = "./"
model_weight_save_path = "./"
num_classes = 4

batch_size = 16
num_workers = 0
lr = 1e-4

total_epoch = 100

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
import torchvision.datasets as datasets
import os
import numpy as np
# Data loading code
traindir = os.path.join(dataset_path, 'train')
testdir = os.path.join(dataset_path, 'test')

resize_train_mean = [0.5065323, 0.5065452, 0.5065475]
resize_train_std = [0.2080959, 0.2081009, 0.20810248]

resize_test_mean = [0.5072302, 0.50723076, 0.5072306]
resize_test_std = [0.20929854, 0.20929676, 0.20929694]
transform_train = transforms.Compose([
    transforms.Resize((224, 224)), # 이미지 resize
    # transforms.RandomCrop(224), # 이미지를 랜덤으로 크롭
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # 이미지 지터링(밝기, 대조, 채비, 색조)
    # transforms.RandomHorizontalFlip(p = 0.5), # p확률로 이미지 좌우반전
    # transforms.RandomVerticalFlip(p = 0.5), # p확률로 상하반전
    transforms.ToTensor(),
    transforms.Normalize(resize_train_mean, resize_train_std)
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(resize_test_mean, resize_test_std)
])

train_dataset = datasets.ImageFolder(traindir, transform_train)
test_dataset = datasets.ImageFolder(testdir, transform_test)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True)


from torchvision.models import efficientnet_v2_s, efficientnet_b0
model = efficientnet_v2_s(num_classes=4).to(device)



CEloss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

import numpy as np
from tqdm import tqdm
total_iteration_per_epoch = int(np.ceil(len(train_dataset)/batch_size))
best_val = 0
for epoch in range(1, total_epoch + 1):
    model.train()
    total = 0
    correct = 0
    for itereation, (input, target) in tqdm(enumerate(train_loader)):
        images = input.to(device)
        labels = target.to(device)

        # Forward pass
        outputs = model(images)
        loss = CEloss(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()
        if itereation %50 ==0:
            print('Epoch [{}/{}], Iteration [{}/{}] Loss: {:.4f} Acc: {:.4f}'.format(epoch, total_epoch, itereation+1, total_iteration_per_epoch, loss.item(), correct/total))
            correct = 0
            total = 0
    if epoch % 10 == 0:
      torch.save(model.state_dict(), model_weight_save_path + 'model_' + str(epoch) + ".pth")

    model.eval()
    with torch.no_grad():
      correct = 0   
      total = 0
      for input, target in test_loader:
          images = input.to(device)
          labels = target.to(device)

          # Forward pass
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += len(labels)
          correct += (predicted == labels).sum().item()

      print('Epoch [{}/{}], Test Accuracy of the model on the {} test images: {} %'.format(epoch, total_epoch, total, 100 * correct / total))
      val = 100 * correct / total
      if best_val < val:
        torch.save(model.state_dict(), model_weight_save_path + 'model_best_' + str(epoch) + ".pth")
        ov_model_cls = ov.convert_model(model)
        ov.save_model(ov_model_cls, f'conv_next.xml')
        best_val = val