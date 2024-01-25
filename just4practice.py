import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 예시 데이터
labels_tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.float32)
labels_list = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

print(f'size of tensor: {labels_tensor.shape}')
# 결과: torch.Size([3, 3])

print(f'size of list: {len(labels_list)}')
# 결과: 3

print()
data_loader_tensor = DataLoader(labels_tensor, batch_size=2)
for batch in data_loader_tensor:
    print(batch.shape)
    print(batch)

print()
# DataLoader 예시
data_loader_list = DataLoader(labels_list, batch_size=2)
for batch in data_loader_list:
    print(len(batch))
    print(batch)
# 결과: 2, 1