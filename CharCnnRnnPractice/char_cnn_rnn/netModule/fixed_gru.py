import torch
import torch.nn as nn

class fixed_gru(nn.Module):
    def __init__(self, emb_dim, num_steps):
        super(fixed_gru).__init__()
        
        # update gate
        self.i2h_update = nn.Linear(emb_dim, emb_dim)
        self.h2h_update = nn.Linear(emb_dim, emb_dim)

        # reset gate
        self.i2h_reset = nn.Linear(emb_dim, emb_dim)
        self.h2h_reset = nn.Linear(emb_dim, emb_dim)

        # candidate hidden state
        self.i2h = nn.Linear(emb_dim, emb_dim)
        self.h2h = nn.Linear(emb_dim, emb_dim)

        self.num_steps = num_steps
    
    def forward(self, txt):
        # usually size of input txt is (batch_size, seq_length, emb_dim).
        res = []
        res_intermediate = []
        for i in range(self.num_steps):
            if i == 0: # first input
                # (batch_size, seq_length, emb_dim) 
                output = torch.tanh(self.i2h(txt[:, i])).unsqueeze(1)
            else:
                # compute update and reset gates
                update = torch.sigmoid(self.i2h_update(txt[:, i])+self.h2h_update(res[i-1]))