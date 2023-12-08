import torch
from torch import nn

hidden = 3

l = [[[1,3,4],[4,3,5],[3,6,3]],
     [[6,4,3],[3,4,5],[6,1,4]],
     [[5,4,5],[3,7,7],[4,7,6]]]
l = torch.Tensor(l)
i2h = nn.Linear(hidden,hidden)
#print(l.size())
#print(l[:,0])

output = i2h(l[:,0])
print(output)