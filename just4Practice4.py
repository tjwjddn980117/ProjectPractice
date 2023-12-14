import torch
scores = torch.tensor([[5,2,1],
                       [5,1,3],
                       [4,5,1]])
max_ids = torch.argmax(scores, dim=1).to('cpu')
print(max_ids)
i = 0
ground_truths = torch.LongTensor(scores.size(0)).fill_(i)
print(ground_truths)

num_correct = (max_ids == ground_truths).sum().item()
print(num_correct)