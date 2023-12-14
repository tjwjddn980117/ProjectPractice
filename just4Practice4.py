import torch

example = torch.tensor([[[1,2,3,4,5],
                         [6,7,8,9,10]], 
                        [[1,2,3,4,5],
                         [6,7,8,9,10]]])

print(example.size())

example = example.permute(0, 2, 1)

print(example.size())
print(example)