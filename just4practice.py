import torch
import torch.nn as nn
import torch.optim as optim

# 예시 데이터 생성
# 입력 데이터 (여기서는 각각 5개의 샘플과 1개의 이진 레이블을 가정)
inputs = torch.randn(5, requires_grad=True)
targets = torch.randint(0, 2, (5,), dtype=torch.float32)

# Binary Cross Entropy Loss 정의
criterion = nn.BCELoss()

# 모델의 출력(inputs)과 실제 레이블(targets) 간의 BCE Loss 계산
loss = criterion(torch.sigmoid(inputs), targets)

# backward pass: 손실에 대한 그래디언트 계산
loss.backward()

# 손실 값 출력
print(inputs)
print([inputs.view(-1)])
print(targets)
print("BCE Loss:", loss.item())