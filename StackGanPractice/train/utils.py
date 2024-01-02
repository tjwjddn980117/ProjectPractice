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
        KLD (float): Loss of KLD.
    '''
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def weights_init(m):
    '''
    This is the function of init the weights of modules.
    
    Inputs:
        m (nn.Module): the module we want to define.
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

def load_params(model, new_param):
    '''
    This is the function for loading the parameters.

    Inputs:
        model (nn.Module): the model already we have.
        new_param ( ): information of loading parameters.
    '''
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    '''
    This is the function for copy Generative model's parameters.

    Inputs:
        model (nn.Module): the model which want to copy.
    
    Outputs:
        flatten (list[]): list of model's parameters.
    '''
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def compute_inception_score(predictions:np, num_splits=1):
    '''
    Calculate the Inception score and return its mean and standard deviation.
    The Inception score is a measure of the quality of the generating model, 
    which gives high diversity and high scores to models that produce lifelike images.

    Inputs:
        predictions (nparray): [batch, num_class].
        Indicates the predicted probability by class 
        obtained by passing through the Inception network.
        num_splits (int): size of split (mini_batch)
    
    Outputs:
        np.mean(scores) (nparray): the mean of kl scores.
        np.std(scores) (nparray): the std of kl scores.
    '''
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        # ex) part = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.2, 0.5, 0.3]])
        part = predictions[istart:iend, :]
        # ex) np.log(part) = [[-2.302, -1.609, -0.357], [-1.204, -0.916, -1.204], [-1.609, -0.693, -1.204]]
        # ex) np.mean(part, 0) = [0.2, 0.367, 0.433] (mean of each class)
        # ex) np.expand_dims = [[0.2, -.367, 0.433]]
        kl = part * \
            (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        # ex) np.sum(kl, 1) = [0.145, 0.046, 0.044]
        # ex) np.mean(np.sum(kl, 1)) = 0.07868
        kl = np.mean(np.sum(kl, 1))
        # scores = [1.08186]
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def negative_log_posterior_probability(predictions, num_splits=1):
    '''
    Negative log post-probability is a measure 
    that gives a high score to a model 
    that accurately predicts a class with a high probability.

    Inputs:
        predictions (nparray): [batch, num_class].
        Indicates the predicted probability by class 
        obtained by passing through the Inception network.
        num_splits (int): size of split (mini_batch)
    
    Outputs:
        np.mean(scores) (nparray): the mean of negative_log scores.
        np.std(scores) (nparray): the std of negative_log scores.
    '''
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)