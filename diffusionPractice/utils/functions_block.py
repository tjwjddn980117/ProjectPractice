import torch
from torch import nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functions import *

# small helper modules
def Upsample(dim, dim_out = None):
    '''
    Inputs:
        dim (int): input dimmension.
        dim_out (bool): choose to out with same dim, or different dim.
    '''
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)
