import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        # d_model = dimension of model
        # max_len = max sequence of d_model
        # device = portable device
        super(PositionalEncoding).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False # positional encoding sould not compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step = 2 ,device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / 10000**(_2i/d_model))
        self.encoding[:, 1::2] = torch.cos(pos / 10000**(_2i/d_model))
    
    def forward(self, x):
        # self.encoing
        # max_len = 512, d_model = 512
        batch_size, seq_len = x.size()
        # batch_size = 128, seq_len = 30

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb [128, 30, 512]