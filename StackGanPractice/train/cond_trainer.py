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
from torch.utils.tensorboard import SummaryWriter

from utils import *

device = torch.device("cuda" if cfg.CUDA else "cpu")

######## This is text_to_image task #######
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, imsize):
        '''
        condition GAN Trainer.

        Arguments:
            output_dir (str): the direction of output.
            data_loader (data_loader): data loader.
        '''
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = SummaryWriter(self.log_dir)
        
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
    
    def prepare_data(self, data):
        '''
        The function to prepare data.

        Inputs:
            data: (tuple):
                imgs (list): list of preprocessing images.
                wrong_imgs (list): list of preprocessing adversarial images.
                embedding (ndarray): embedding of one caption. [emb_dim].
                key (str): name of imgs.
        
        Outputs:
            imgs (list[imgs]): list of images that work on cpu.
            real_vimgs (list[imgs]): list of real images that work on gpus.
            wrong_vimgs (list[imgs]): list of wrong images that work on gpus.
            vembedding (nparray): embedding of one caption work on gpus. 
        '''

        imgs, w_imgs, t_embedding, _ = data
    
        real_vimgs, wrong_vimgs = [], []
    
        vembedding = t_embedding.to(device)
    
        for i in range(self.num_Ds):
            real_vimgs.append(imgs[i].to(device))
            wrong_vimgs.append(w_imgs[i].to(device))
    
        return imgs, real_vimgs, wrong_vimgs, vembedding

    def train_Dnet(self, idx, count):
        '''
        training the Discriminate model.

        Inputs:
            idx (int): number of index.
            count (int): number of training count.
        
        Outputs:
            errD (float): sum of error about errD_real+errD_wrong+errD_fake with some lambda. BCELoss.
        '''
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)
        # self.mu come from netG. 
        # mu (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]. just x[:, :self.ef_dim] after fc.
        criterion, mu = self.criterion, self.mu

        netD, optD = self.netsD[idx], self.optimizersD[idx]
        real_imgs = self.real_imgs[idx]
        wrong_imgs = self.wrong_imgs[idx]
        fake_imgs = self.fake_imgs[idx]

        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]
        #
        netD.zero_grad()

        # mu doesn't learning, and it just use with coping.  
        # becuase we use 'mu' from 'self.txt_embedding', and 'self.txt_embedding' is saved information.
        # so, we have to keep this imformation while we learning.
        # then, 'mu.detach()' doesn't learning, 
        #  but useage in netD(encoder_mu, encoder_logvar, c_code) should learning.
        real_logits = netD(real_imgs, mu.detach())
        wrong_logits = netD(wrong_imgs, mu.detach())
        fake_logits = netD(fake_imgs.detach(), mu.detach())

        errD_real = criterion(real_logits[0], real_labels)
        errD_wrong = criterion(wrong_logits[0], fake_labels)
        errD_fake = criterion(fake_logits[0], fake_labels)

        if len(real_logits) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
            errD_real_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion(real_logits[1], real_labels)
            errD_wrong_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion(wrong_logits[1], real_labels)
            errD_fake_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion(fake_logits[1], fake_labels)
            #
            errD_real = errD_real + errD_real_uncond
            errD_wrong = errD_wrong + errD_wrong_uncond
            errD_fake = errD_fake + errD_fake_uncond
            #
            errD = errD_real + errD_wrong + errD_fake
        else:
            errD = errD_real + 0.5 * (errD_wrong + errD_fake)
        # backward
        errD.backward()
        # update parameters
        optD.step()

        # log
        if flag == 0:
            summary_D = summary.scalar('D_loss%d' % idx, errD.data[0])
            self.summary_writer.add_summary(summary_D, count)
        return errD
    
    def train_Gnet(self, count):
        '''
        training the Generation model.

        Inputs:
            count (int): number of training count.
        
        Outputs:
            errG_total (float): sum of error about errG+like_mu1+like_cov1. errG is BCELoss and others are MSE.
        '''
        self.netG.zero_grad()
        errG_total = 0
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)
        # self.mu come from netG. 
        # mu (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]. just x[:, :self.ef_dim] after fc.
        criterion, mu = self.criterion, self.mu
        real_labels = self.real_labels[:batch_size]

        for i in range(self.num_Ds):
            netD = self.netsD[i]
            outputs = netD(self.fake_imgs[i])
            errG = criterion(outputs[0], real_labels)