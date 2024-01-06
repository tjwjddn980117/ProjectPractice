import torch
import torch.nn as nn
import torch.optim as optim
import math

# 예시 데이터 생성
# 입력 데이터 (여기서는 각각 5개의 샘플과 1개의 이진 레이블을 가정)
inputs = torch.tensor([[0,2*math.log(2),8*math.log(1)],[0,8*math.log(2),18*math.log(2)]])

print(inputs)

print(inputs.mul(0.5))

std = inputs.mul(0.5).exp_()

print(std)

eps = torch.randn_like(std)

print(eps)