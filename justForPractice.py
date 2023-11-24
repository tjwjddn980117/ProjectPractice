import torch
import torch.nn as nn

softmax = nn.Softmax(dim=-1)
x = torch.randn(2, 3, 3)
output = softmax(x)

print("Input:", x)
print("Output:", output)