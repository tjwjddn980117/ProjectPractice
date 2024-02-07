import torch
import torch.nn as nn

# [batch_size, number_of_class, number_of_sequence]
# [batch_size, number_of_sequence] (label is the number of class -1)
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
