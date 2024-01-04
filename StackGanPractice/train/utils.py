from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
from copy import deepcopy

from miscc.config import cfg

from tensorboard import summary
from tensorboard import FileWriter

from ..models.generation import G_NET
from ..models.discriminator import D_NET64, D_NET128, D_NET256, D_NET512, D_NET1024
from ..models.utils import INCEPTION_V3

def compute_mean_covariance(img):
    '''
    This is the function for compute mean and covariance of image.
    
    Inputs:
        img (nparray): [B, C, H, W]. original image.
    
    Outputs:
        mu (nparray): [B, C, 1, 1]. \n
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
        mu (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]. \n
        logvar (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM].

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
        model (nn.Module): the model already we have. \n
        new_param (parameter): information of loading parameters.
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
        obtained by passing through the Inception network. \n
        num_splits (int): size of split (mini_batch).
    
    Outputs:
        np.mean(scores) (nparray): the mean of kl scores. \n
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
        obtained by passing through the Inception network. \n
        num_splits (int): size of split (mini_batch).
    
    Outputs:
        np.mean(scores) (nparray): the mean of negative_log scores. \n
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

def load_network(gpus):
    '''
    The function that loading models.
    We get models from cfg files.

    Inputs:
        gpus: afordable gpus.

    Outputs:
        netG (nn.Module): Generation model. 
        netsD (list[nn.Module]): Discrimination model with Branches.
        len(netsD) (int): number of discrimination models.
        inception_model (nn.Module): INCEPTION_V3 pre-trained model.
        count (int): number of training count. 
    '''
    #just init G_NET.
    netG = G_NET()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    print(netG)

    # it's getting deeper, D_NET's size getting bigger.
    netsD = []
    if cfg.TREE.BRANCH_NUM > 0:
        netsD.append(D_NET64())
    if cfg.TREE.BRANCH_NUM > 1:
        netsD.append(D_NET128())
    if cfg.TREE.BRANCH_NUM > 2:
        netsD.append(D_NET256())
    if cfg.TREE.BRANCH_NUM > 3:
        netsD.append(D_NET512())
    if cfg.TREE.BRANCH_NUM > 4:
        netsD.append(D_NET1024())
    # TODO: if cfg.TREE.BRANCH_NUM > 5:

    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)
        # print(netsD[i])
    print('# of netsD', len(netsD))

    count = 0
    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_G)

        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        count = cfg.TRAIN.NET_G[istart:iend]
        count = int(count) + 1

    if cfg.TRAIN.NET_D != '':
        for i in range(len(netsD)):
            print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
            state_dict = torch.load('%s%d.pth' % (cfg.TRAIN.NET_D, i))
            netsD[i].load_state_dict(state_dict)

    inception_model = INCEPTION_V3()

    if cfg.CUDA:
        netG.cuda()
        for i in range(len(netsD)):
            netsD[i].cuda()
        inception_model = inception_model.cuda()
    inception_model.eval()

    return netG, netsD, len(netsD), inception_model, count

def define_optimizers(netG, netsD):
    '''
    defining optimizers in each of netG and netsD.
    
    Inputs:
        netG (nn.Module): generation model.
        netsD (list[nn.Module]): list of discrimination models.

    Outputs:
        optimizerG (optim): optimizer for generation model.
        optimizersD (list[optim]): optimizer for discriminator models.
    '''
    optimizersD = []
    num_Ds = len(netsD)
    for i in range(num_Ds):
        opt = optim.Adam(netsD[i].parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
        optimizersD.append(opt)

    # G_opt_paras = []
    # for p in netG.parameters():
    #     if p.requires_grad:
    #         G_opt_paras.append(p)
    optimizerG = optim.Adam(netG.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            betas=(0.5, 0.999))
    return optimizerG, optimizersD


def save_model(netG, avg_param_G, netsD, epoch, model_dir):
    '''
    the function that save the model.

    Inputs:
        netG (nn.Module): the generation model already we have.
        avg_param_G (parameter): information of loading parameters of generation.
        netsD (list[nn.Module]): the list of discrimination model already we have.
        epoch (int): the number of epochs we have trained.
        model_dir (str): the path of saving way.
    '''
    load_params(netG, avg_param_G)
    torch.save(
        netG.state_dict(),
        '%s/netG_%d.pth' % (model_dir, epoch))
    for i in range(len(netsD)):
        netD = netsD[i]
        torch.save(
            netD.state_dict(),
            '%s/netD%d.pth' % (model_dir, i))
    print('Save G/Ds models.')

def save_img_results(imgs_tcpu, fake_imgs, num_imgs,
                     count, image_dir, summary_writer):
    '''
    saving results of image.
    
    Inputs:
        imgs_tcpu ( ): list of real images.
        fake_imgs (list): list of fake images.
        num_imgs (int): the number of images.
        count (int): number of training count.
        image_dir (str): the path of saving image.
        summary_writer ( ): maybe for the tensorboard.
    '''
    num = cfg.TRAIN.VIS_COUNT

    # The range of real_img (i.e., self.imgs_tcpu[i][0:num]).
    # is changed to [0, 1] by function vutils.save_image.
    real_img = imgs_tcpu[-1][0:num]
    vutils.save_image(
        real_img, '%s/real_samples.png' % (image_dir),
        normalize=True)
    real_img_set = vutils.make_grid(real_img).numpy()
    real_img_set = np.transpose(real_img_set, (1, 2, 0))
    real_img_set = real_img_set * 255
    real_img_set = real_img_set.astype(np.uint8)
    sup_real_img = summary.image('real_img', real_img_set)
    summary_writer.add_summary(sup_real_img, count)

    for i in range(num_imgs):
        fake_img = fake_imgs[i][0:num]
        # The range of fake_img.data (i.e., self.fake_imgs[i][0:num])
        # is still [-1. 1]...
        vutils.save_image(
            fake_img.data, '%s/count_%09d_fake_samples%d.png' %
            (image_dir, count, i), normalize=True)

        fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()

        fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
        fake_img_set = (fake_img_set + 1) * 255 / 2
        fake_img_set = fake_img_set.astype(np.uint8)

        sup_fake_img = summary.image('fake_img%d' % i, fake_img_set)
        summary_writer.add_summary(sup_fake_img, count)
        summary_writer.flush()
