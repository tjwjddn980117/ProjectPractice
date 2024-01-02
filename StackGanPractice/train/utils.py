from __future__ import print_function

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import os
import time
from PIL import Image, ImageFont, ImageDraw
from copy import deepcopy

from miscc.config import cfg
from miscc.utils import mkdir_p

from tensorboard import summary
from tensorboard import FileWriter

from ..models.generation import G_NETrr
from ..models.discriminator import D_NET64, D_NET128, D_NET256, D_NET512, D_NET1024
from ..models.utils import INCEPTION_V3

def compute_mean_covariance(img):
    '''
    This is the function for compute mean and covariance of image.
    
    Inputs:
        img (nparray): [B, C, H, W]. original image.
    
    Outputs:
        mu (nparray): [B, C, 1, 1] .
        covariance (nparray): [B, C, C]. covariance for channels.
    '''
    batch_size = img.size(0)
    channel_num = img.size(1)
    height = img.size(2)
    width = img.size(3)
    num_pixels = height * width

    # batch_size * channel_num * 1 * 1
    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)

    # batch_size * channel_num * num_pixels
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.view(batch_size, channel_num, num_pixels)
    # batch_size * num_pixels * channel_num
    img_hat_transpose = img_hat.transpose(1, 2)
    # batch_size * channel_num * channel_num
    # it's cause pixel^2
    covariance = torch.bmm(img_hat, img_hat_transpose)
    covariance = covariance / num_pixels

    return mu, covariance

def KL_loss(mu, logvar):
    '''
    This is the function for calculate KLD loss.
    In this function, we use the KLD in auto-encoder.
    It's KL(q(z|x)||p(z)) = E[log q(z|x) - log p(z)].
    
    Inputs:
        mu (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]
        logvar (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]

    Outputs:
        KLD ( ):
    '''
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def weights_init(m):
    '''
    This is the function of init the weights of modules.
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
