import torch
from torch import nn
from einops import repeat
from ..utils.helpers import *
from einops.layers.torch import Rearrange

class Upsample(nn.Module):
    '''
    Arguments:
        dim (int): input dimmension.
        dim_out (bool): choose to out with same dim, or different dim.
        factor (int): upsampling size. you can think about the size.

    Inputs:
        x (tensor): [B, C, H, W]
    
    Outputs:
        x (tensor): [B, C, factor*H, factor*W]
    '''
    def __init__(self, dim, dim_out = None,factor = 2):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        '''
        Initing weight with [B, C, H, W]
        '''
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        # In this case, we have the same weight in units of factor_squared.
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(dim, dim_out = None, factor = 2):
    '''
    Inputs:
        dim (int): input dimmension.
        dim_out (bool): choose to out with same dim, or different dim. 
        factor (int): upsampling size. you can think about the size.
    
    Outputs:
        nn.Sequential(Rearrange[b,c,2h,2w]->[b,4c,h,w] -> nn.Conv2d)
        input: [b,c,2h,2w]
        output:[b,4c,h,w]
    '''
