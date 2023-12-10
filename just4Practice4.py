import torch
cls_id = torch.randint(5, (1,))

print(cls_id)

diagonal = torch.tensor([[1], [2], [1]])

scores = torch.tensor([[3, 3, 3],
                       [3, 3, 3],
                       [3, 3, 3]])

print(diagonal.expand_as(scores))