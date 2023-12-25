import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo

from utils import *

# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    '''
    This block work for up-sampling.
    
    Inputs:
        [B, in_planes, H, W]
    Outputs:
        [B, out_planes, H*2, W*2]
    '''
    block = nn.Sequential(
        # [B, in_planes, H, W]
        nn.Upsample(scale_factor=2, mode='nearest'),
        # [B, in_planes, H*2, W*2]
        conv3x3(in_planes, out_planes * 2),
        # [B, out_planes*2, H*2, W*2]
        nn.BatchNorm2d(out_planes * 2),
        # [B, out_planes*2, H*2, W*2]
        GLU()
        # [B, out_planes, H*2, W*2]
    )
    return block

# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    '''
    This block work for Relu.
    
    Inputs:
        [B, in_planes, H, W]
    Outputs:
        [B, out_planes, H, W]
    '''
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class ResBlock(nn.Module):
    '''
    This block work for ResBlock.
    
    Inputs:
        [B, channel_num, H, W]
    Outputs:
        [B, channel_num, H, W]
    '''
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )


    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    '''
    Conditioning Augumentation Network. 
    This is a key idea that enables backpropagation using the Reparametricization technique of VAEs.

    Inputs:
        [batch_size, cfg.TEXT.DIMENSION]

    Returns:
        [batch_size, cfg.GAN.EMBEDDING_DIM], 
        [batch_size, cfg.GAN.EMBEDDING_DIM],
        [batch_size, cfg.GAN.EMBEDDING_DIM].
    '''
    def __init__(self):
        super(CA_NET).__init__()
        self.t_dim = cfg.TEXT.DIMENSION
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        '''
        In encode part, we define mu and logvar randomly.
        We Initialize with 'parameters to be averaged' and 'parameters to be dispersed'
        and then update these parameters to values that naturally represent 
        mean and variance during the learning process.
        
        Inputs:
            [batch_size, cfg.TEXT.DIMENSION].

        Outputs:
            [batch_size, cfg.GAN.EMBEDDING_DIM], [batch_size, cfg.GAN.EMBEDDING_DIM].
        '''
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar
    
    def reparametrize(self, mu, logvar):
        '''
        The reparametrize method samples latent vectors using mean and variance. 
        This is a key idea that enables backpropagation using the Reparametricization technique of VAEs.
        Inputs:
            [batch_size, cfg.GAN.EMBEDDING_DIM], [batch_size, cfg.GAN.EMBEDDING_DIM].
        Outputs:
            [batch_size, cfg.GAN.EMBEDDING_DIM]
        '''
        # Divide the log var by 2 
        #  and apply an exponential function to calculate the standard deviation (std).
        # std = [batch_size, cfg.GAN.EMBEDDING_DIM]
        std = logvar.mul(0.5).exp_()
        # eps = [batch_size, cfg.GAN.EMBEDDING_DIM]
        if cfg.CUDA:
             eps = torch.randn_like(std).to('cuda')
        else:
             eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar
    
class INIT_STAGE_G(nn.Module):
    '''
    This is the code about initialize G_Stage
    '''
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if cfg.GAN.B_CONDITION:
            self.in_dim = cfg.GAN.Z_DIM + cfg.GAN.EMBEDDING_DIM
        else:
            self.in_dim = cfg.GAN.Z_DIM
        self.define_module()
    
    def define_module(self):
        '''
        this is the model of defining module
        '''
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU()
        )
        
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
    
    def forward(self, z_code, c_code=None):
        if cfg.GAN.B_CONDITION and c_code is not None:
            # in_code size is [B, (c_code + z_code)]
            in_code = torch.cat((c_code, z_code), 1)
        else:
            # in_code size is [B, z_code]
            in_code = z_code

        # [B, in_code]
        out_code = self.fc(in_code)
        # [B, [ngf x 4 x 4]]
        # However, we will assume that ngf is 16ngf for easy calculation.
        # so, the size will [B, [16ngf x 4 x 4]]
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # [B, 16ngf, 4, 4]
        out_code = self.upsample1(out_code)
        # [B, 8ngf, 8, 8]
        out_code = self.upsample2(out_code)
        # [B, 4ngf, 16, 16]
        out_code = self.upsample3(out_code)
        # [B, 2ngf, 32, 32]
        out_code = self.upsample4(out_code)
        # [B, ngf, 64, 64]

        return out_code
    
class NEXT_STAGE_G(nn.Module):
    '''
    This is the model of next stage of Generation.
    '''
    def __init__(self, ngf, num_residual=cfg.GAN.R_NUM):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if cfg.GAN.B_CONDITION:
            self.ef_dim = cfg.GAN.EMBEDDING_DIM
        else:
            self.ef_dim = cfg.GAN.Z_DIM
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        '''
        This is the function for make layer.
        The layer don't change the size.
        '''
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        '''
        This is the model of defining module of layers.
        '''
        ngf = self.gf_dim
        efg = self.ef_dim
        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = upBlock(ngf, ngf // 2)
    
    def forward(self, h_code, c_code):
        # h_code will the data of image file 
        s_size = h_code.size(2)
        # c_code will resize the text to image
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)
        # state size (ngf+egf) x in_size x in_size
        h_c_code = torch.cat((c_code, h_code), 1)
        # state size ngf x in_size x in_size
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code
    
    