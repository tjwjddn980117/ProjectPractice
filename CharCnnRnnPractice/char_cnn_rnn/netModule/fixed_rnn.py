import torch
from torch import nn

class fixed_rnn(nn.Module):
    def __init__(self, emb_dim, num_steps):
        super(fixed_rnn).__init__()
        self.i2h = nn.Linear(emb_dim, emb_dim)
        self.h2h = nn.Linear(emb_dim, emb_dim)
        self.relu = nn.ReLU()
        self.num_steps = num_steps
    
    def forward(self, txt):
        res = []
        for i in range(self.num_steps):
            i2h = self.i2h(txt[:, i]).unsqueeze(1)
            if i == 0:
                output = self.relu(i2h)
            else:
                h2h = self.h2h(res[i-1])
                output = self.relu(i2h + h2h)
            
            res.append(output)
        
        res = torch.cat(res, dim=1)
        res = torch.mean(res, dim=1)
        return res