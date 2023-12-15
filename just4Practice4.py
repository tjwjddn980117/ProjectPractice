import torch
scores = torch.tensor([[4,2,6,1],
                       [6,8,3,1],
                       [7,4,2,9]])
matches = torch.tensor([[1,0,3,2],
                        [0,1,3,2],
                        [2,3,1,0]])
for i, s in enumerate(scores): # for each class
        # inds is the index that before sorted
        _, inds = torch.sort(s, descending=True)
        print(inds)
        matches[i] = matches[i, inds]
    

print(matches)