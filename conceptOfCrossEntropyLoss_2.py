import torch
import torch.nn as nn

output = torch.Tensor(
    [
        [0, 1, 0, 0, 0]
    ]
)

# 클래스 인덱스로 변환
target = torch.LongTensor([1])

criterion = nn.CrossEntropyLoss()

loss = criterion(output, target)

print(loss)