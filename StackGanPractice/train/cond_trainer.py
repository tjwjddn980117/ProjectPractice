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
            KL_loss (float): KL_loss
            errG_total (float): sum of error about kl_loss+errG+like_mu1+like_cov1. 
            errG is BCELoss and kl_loss is KLLoss others are MSE.
        '''
        self.netG.zero_grad()
        errG_total = 0
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)
        # self.mu come from netG. 
        # mu (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]. just x[:, :self.ef_dim] after fc.
        criterion, mu, logvar = self.criterion, self.mu, self.logvar
        real_labels = self.real_labels[:batch_size]

        for i in range(self.num_Ds):
            netD = self.netsD[i]
            # fake_imgs will be created.
            outputs = netD(self.fake_imgs[i])
            errG = criterion(outputs[0], real_labels)
            errG_total = errG_total + errG

            if flag == 0:
                summary_G = summary.scalar('G_loss%d' % i, errG.data[0])
                self.summary_writer.add_summary(summary_G, count)
        
        # Compute color preserve losses
        if cfg.TRAIN.COEFF.COLOR_LOSS > 0:
            # Size getting bigger and not changing the mood of the painting.
            # so, we sould check fake_imgs[-1](lagest size) 
            #  and the things just before (fake_imgs[-2],...)
            if self.num_Ds > 1:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-1])
                mu2, covariance2 = \
                    compute_mean_covariance(self.fake_imgs[-2].detach())
                like_mu2 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov2 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
                    nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu2 + like_cov2
            if self.num_DS > 2:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-2])
                mu2, covariance2 = \
                    compute_mean_covariance(self.fake_imgs[-3].detach())
                like_mu1 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov1 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
                    nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu1 + like_cov1

        if flag == 0:
                sum_mu = summary.scalar('G_like_mu2', like_mu2.data[0])
                self.summary_writer.add_summary(sum_mu, count)
                sum_cov = summary.scalar('G_like_cov2', like_cov2.data[0])
                self.summary_writer.add_summary(sum_cov, count)
                if self.num_Ds > 2:
                    sum_mu = summary.scalar('G_like_mu1', like_mu1.data[0])
                    self.summary_writer.add_summary(sum_mu, count)
                    sum_cov = summary.scalar('G_like_cov1', like_cov1.data[0])
                    self.summary_writer.add_summary(sum_cov, count)

        kl_loss = KL_loss(mu, logvar) * cfg.TRAIN.COEFF.KL
        errG_total = errG_total + kl_loss
        errG_total.backward()
        self.optimizerG.step()
        return kl_loss, errG_total

    def train(self):
        self.netG, self.netsD, self.num_Ds,\
                self.inception_model, start_count = load_network(self.gpus)
        avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizersD = \
            define_optimizers(self.netG, self.netsD)

        self.criterion = nn.BCELoss()

        self.real_labels = torch.full((self.batch_size, ), 1.0, device=device)
        self.fake_labels = torch.full((self.batch_size, ), 0.0, device=device)
        nz = cfg.GAN.Z_DIM
        self.gradient_one = torch.FloatTensor([1.0]).to(device)
        self.gradient_half = torch.FloatTensor([0.5]).to(device)
        noise = torch.FloatTensor(self.batch_size, nz).to(device)
        fixed_noise = torch.FloatTensor(self.batch_size, nz).normal_(0, 1).to(device)
        
        predictions = []
        count = start_count
        # we may stop training.
        start_epoch = start_count // (self.num_batches) 
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            for step, data in enumerate(self.data_loader):
                self.imgs_tcpu, self.real_imgs, self.wrong_imgs, self.txt_embedding \
                                                        = self.prepare_data(data)
                
                #######################################################
                # (1) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, self.mu, self.logvar = self.netG(noise, self.txt_embedding)

                #######################################################
                # (2) Update D network
                ######################################################
                errD_total = 0
                for i in range(self.num_Ds):
                    errD = self.train_Dnet(i, count)
                    errD_total += errD

                #######################################################
                # (3) Update G network: maximize log(D(G(z)))
                ######################################################
                kl_loss, errG_total = self.train_Gnet(count)
                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                # for inception score
                pred = self.inception_model(self.fake_imgs[-1].detach())
                predictions.append(pred.data.cpu().numpy())

                if count % 100 == 0:
                    summary_D = summary.scalar('D_loss', errD_total.data[0])
                    summary_G = summary.scalar('G_loss', errG_total.data[0])
                    summary_KL = summary.scalar('KL_loss', kl_loss.data[0])
                    self.summary_writer.add_summary(summary_D, count)
                    self.summary_writer.add_summary(summary_G, count)
                    self.summary_writer.add_summary(summary_KL, count)

                count = count + 1

                if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                    save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
                    # Save images
                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)
                    #
                    self.fake_imgs, _, _ = \
                        self.netG(fixed_noise, self.txt_embedding)
                    save_img_results(self.imgs_tcpu, self.fake_imgs, self.num_Ds,
                                     count, self.image_dir, self.summary_writer)
                    #
                    load_params(self.netG, backup_para)

                    # Compute inception score
                    if len(predictions) > 500:
                        predictions = np.concatenate(predictions, 0)
                        mean, std = compute_inception_score(predictions, 10)
                        # print('mean:', mean, 'std', std)
                        m_incep = summary.scalar('Inception_mean', mean)
                        self.summary_writer.add_summary(m_incep, count)
                        #
                        mean_nlpp, std_nlpp = \
                            negative_log_posterior_probability(predictions, 10)
                        m_nlpp = summary.scalar('NLPP_mean', mean_nlpp)
                        self.summary_writer.add_summary(m_nlpp, count)
                        #
                        predictions = []

            end_t = time.time()
            print('''[%d/%d][%d]
                         Loss_D: %.2f Loss_G: %.2f Loss_KL: %.2f Time: %.2fs
                      '''  # D(real): %.4f D(wrong):%.4f  D(fake) %.4f
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.data[0], errG_total.data[0],
                     kl_loss.data[0], end_t - start_t))

        save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
        self.summary_writer.close()

    def save_superimages(self, images_list, filenames, save_dir, split_dir, imsize):
        '''
        This function save multiple images at once.

        Inputs:r
            images_list (list[Tensor]): Tensor of images to be saved.
            folder (str): Path to the folder where images will be saved.
            startID (int): Starting ID used in the image filenames.
            imsize (int): Size of image. 
        '''
        batch_size = images_list[0].size(0)
        num_sentences = len(images_list)
        for i in range(batch_size):
            s_tmp = '%s/super/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            #
            savename = '%s_%d.png' % (s_tmp, imsize)
            super_img = []
            for j in range(num_sentences):
                img = images_list[j][i]
                # print(img.size())
                img = img.view(1, 3, imsize, imsize)
                # print(img.size())
                super_img.append(img)
                # break
            super_img = torch.cat(super_img, 0)
            vutils.save_image(super_img, savename, nrow=10, normalize=True)

    def save_singleimages(self, images, filenames, save_dir, split_dir, sentenceID, imsize):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d_sentence%d.png' % (s_tmp, imsize, sentenceID)
            # range from [-1, 1] to [0, 255]
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)
