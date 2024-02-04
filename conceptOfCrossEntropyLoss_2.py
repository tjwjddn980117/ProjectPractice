import torch
import torch.nn as nn

output = torch.Tensor(
    [[
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0]
    ],
    [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0]
    ]]
)

target = torch.LongTensor([[1,2,3],[1,2,3]])

output = output.transpose(1,2).contiguous()
criterion = nn.CrossEntropyLoss()
print(output.shape)
print(target.shape)
loss = criterion(output, target)

print(loss)
