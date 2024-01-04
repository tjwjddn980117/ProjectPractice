from __future__ import print_function

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import os
import time
from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p

from tensorboard import summary
from tensorboard import FileWriter

from utils import *

class GANTrainer(object):
    def __init__(self, output_dir, data_loader, imsize):
        '''
        GAN Trainer.

        Arguments:
            output_dir: the direction of output.
        '''
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

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
            data (list[imgs]): list of images.
        
        Outputs:
            imgs (list[imgs]): list of images that work on cpu.
            vimgs (list[imgs]): list of images that work on gpus.
        '''
        imgs = data
        vimgs = []
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        for i in range(self.num_Ds):
            vimgs.append(imgs[i].to(device))
            return imgs, vimgs
    
    def train_Dnet(self, idx, count):
        '''
        training the Discriminate model.

        Inputs:
            idx (int): number of index.
            count (int): number of training count.
        
        Outputs:
            errD (float): sum of error about errD_real+errD_fake. BCELoss.
        '''
        flag = count % 100
        # if real_imgaes already cut for batches?
        # real_imgs [size_info, B, C, H, W]
        batch_size = self.real_imgs[0].size(0)
        criterion = self.criterion

        netD, optD = self.netsD[idx], self.optimizersD[idx]
        real_imgs = self.real_imgs[idx]
        fake_imgs = self.fake_imgs[idx]
        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]
        #
        netD.zero_grad()
        # outputs [B], it pass the sigmoid(0 < x < 1)
        # real_logits should be [tensor([...])]
        # so, we should re make our data real_logits[0], fake_logits[0].
        real_logits = netD(real_imgs)
        fake_logits = netD(fake_imgs.detach())
        # calculate error rates.
        errD_real = criterion(real_logits[0], real_labels)
        errD_fake = criterion(fake_logits[0], fake_labels)
        # sum discriminate error rates.
        errD = errD_real + errD_fake
        errD.backward()
        # update parameters
        optD.step()
        # print the logs.
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
        # if real_imgaes already cut for batches?
        # real_imgs [size_info, B, C, H, W]
        batch_size = self.real_imgs[0].size(0)
        criterion = self.criterion
        real_labels = self.real_labels[:batch_size]

        for i in range(self.num_Ds):
            netD = self.netsD[i]
            # the size of fake_imgs[i] are already prepared by arranged by size.
            # outputs [B], it pass the sigmoid(0 < x < 1)
            # real_logits should be [tensor([...])]
            # so, we should re make our data real_logits[0], fake_logits[0].
            outputs = netD(self.fake_imgs[i])
            errG = criterion(outputs[0], real_labels)
            # errG = self.stage_coeff[i] * errG
            errG_total = errG_total + errG
            if flag == 0:
                summary_G = summary.scalar('G_loss%d' % i, errG.data[0])
                self.summary_writer.add_summary(summary_G, count)

        # Compute color preserve losses
        if cfg.TRAIN.COEFF.COLOR_LOSS > 0:
            # Being bigger and not changing the mood of the painting.
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
            if self.num_Ds > 2:
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

        errG_total.backward()
        self.optimizerG.step()
        return errG_total
    
    def train(self):
        self.netG, self.netsD, self.num_Ds,\
            self.inception_model, start_count = load_network(self.gpus)
        avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizersD = \
            define_optimizers(self.netG, self.netsD)

        self.criterion = nn.BCELoss()

        self.real_labels = torch.FloatTensor(self.batch_size).fill_(1)
        self.fake_labels = torch.FloatTensor(self.batch_size).fill_(0)
        nz = cfg.GAN.Z_DIM
        noise = torch.FloatTensor(self.batch_size, nz)
        fixed_noise = torch.FloatTensor(self.batch_size, nz).normal_(0, 1)
        
        # we can use gpu.
        if cfg.CUDA: 
            self.criterion.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()

        predictions = []
        count = start_count
        # we may stop training.
        start_epoch = start_count // (self.num_batches) 
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            for step, data in enumerate(self.data_loader, 0):
                #######################################################
                # (0) Prepare training data
                ######################################################
                self.imgs_tcpu, self.real_imgs = self.prepare_data(data)

                #######################################################
                # (1) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                self.fake_imgs, _, _ = self.netG(noise)

                #######################################################
                # (2) Update D network
                #     we have backward in 'train_Dnet'
                ######################################################
                errD_total = 0
                for i in range(self.num_Ds):
                    # we already have real_imgs and fake_imgs.
                    errD = self.train_Dnet(i, count)
                    # error of Discriminator
                    errD_total += errD

                #######################################################
                # (3) Update G network: maximize log(D(G(z)))
                #     we have backward in 'train_Gnet'
                ######################################################
                # error of Generator
                errG_total = self.train_Gnet(count)

                # we normalize the netG parameters.
                # avg_param_G is the parameters that copied before training.
                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                # for inception score
                # this is for double checking that it could make the right picture.
                # The moving average thus implemented can be interpreted as applying 99.9% 
                #  to the previous parameter value and 0.1% to the current parameter value. 
                # This way, the parameters of the model can be stabilized and updated smoothly.
                pred = self.inception_model(self.fake_imgs[-1].detach())
                predictions.append(pred.data.cpu().numpy())

                if count % 100 == 0:
                    summary_D = summary.scalar('D_loss', errD_total.data[0])
                    summary_G = summary.scalar('G_loss', errG_total.data[0])
                    self.summary_writer.add_summary(summary_D, count)
                    self.summary_writer.add_summary(summary_G, count)
                if step == 0:
                    print('''[%d/%d][%d/%d] Loss_D: %.2f Loss_G: %.2f'''
                           % (epoch, self.max_epoch, step, self.num_batches,
                              errD_total.data[0], errG_total.data[0]))
                count = count + 1
                
                # saving point.
                if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                    save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
                    save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
                    # Save images
                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)
                    #
                    self.fake_imgs, _, _ = self.netG(fixed_noise)
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
            print('Total Time: %.2fsec' % (end_t - start_t))

        save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
        save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)

        self.summary_writer.close()

    def save_superimages(self, images, folder, startID, imsize):
        '''
        This function save multiple images at once.

        Inputs:
            images (Tensor): Tensor of images to be saved.
            folder (str): Path to the folder where images will be saved.
            startID (int): Starting ID used in the image filenames.
            imsize (int): Size of image.
        '''
        fullpath = '%s/%d_%d.png' % (folder, startID, imsize)
        vutils.save_image(images.data, fullpath, normalize=True)

    def save_singleimages(self, images, folder, startID, imsize):
        '''
        This function individually saves each image. 
        The function transforms the range of image values from [-1, 1] to [0, 1], 
        converts them to integer values between 0 and 255, and saves the images.

        Inputs:
            images (Tensor): Tensor of images to be saved.
            folder (str): Path to the folder where images will be saved.
            startID (int): Starting ID used in the image filenames.
            imsize (int): Size of image.
        '''
        for i in range(images.size(0)):
            fullpath = '%s/%d_%d.png' % (folder, startID + i, imsize)
            # range from [-1, 1] to [0, 1]
            img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    