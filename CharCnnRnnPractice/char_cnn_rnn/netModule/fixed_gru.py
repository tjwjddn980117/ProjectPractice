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
        for i in range(self.num_steps):
            if i == 0: # first input
                # (batch_size, seq_length, emb_dim) change while passing txt[:, i] (batch_size, emb_dim)
                # so, we need to resize (batch_size, emb_dim) to (batch_size, seq_length, emb_dim)
                # that's the reason why we unsqueeze(1)
                output = torch.tanh(self.i2h(txt[:, i])).unsqueeze(1)
            else:
                # compute update and reset gates
                # res[i-1] is the latest (batch_size, 1, emb_dim)
                update = torch.sigmoid(self.i2h_update(txt[:, i])+self.h2h_update(res[i-1]))
                reset = torch.sigmoid(self.i2h_reset(txt[:,i])+self.h2h_reset(res[i-1]))
                
                # compute hidden gate
                hidden_gate = reset*res[i-1]
                p1 = self.i2h(txt[:,i])
                p2 = self.h2h(hidden_gate)
                hidden_cand = torch.tanh(p1 + p2)

                # use gates to interpolate hidden state
                zh = update * hidden_cand
                zhm1 = (1 + (update * -1)) * res[i-1]
                output = zh + zhm1

            
            res.append(output)
                    
        # concat with column way.
        # a,b = tensor(2,3)
        # torch.cat([a,b], dim=1) >> tensor(2,6)
        # torch.cat([a,b], dim=0) >> tensor(4,3)
        res = torch.cat(res, dim=1)
        res = torch.mean(res, dim=1)
        return res