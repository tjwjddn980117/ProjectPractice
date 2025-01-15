import yaml
import argparse
import torch
import cv2
import random
import os
import numpy as np
from tqdm import tqdm
from model.discrete_vae import DiscreteVAE
from torch.utils.data.dataloader import DataLoader
from datasets.mnist_color_texture_dataset import MnistVisualLanguageDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_for_one_epoch(epoch_idx, model, mnist_loader, optimizer, crtierion, config):
    '''
    train with each epoch. 

    Inputs:
        epoch_idx (int): the index of epoch. 
        model (nn.Module): the model for training. 
        mnist_loader (Datasets): the dataset for training. 
        optimizer (optimizer): the optimizer for training. 
        crtierion (crtierion): the crtierion for training. 
        config (dict): the config for training. 
    
    Outputs:
        losses (float): the mean of losses. each losses are from each batch(picture). 
.    '''
    losses = []
    count = 0
    for data in tqdm(mnist_loader):
        # For vae we only need images
        im = data['image']
        im = im.float().to(device)
        optimizer.zero_grad()

        output, kl, log_qy = model(im)
        if config['train_params']['save_vae_training_image'] and count % 25 == 0:
            im_input = cv2.cvtColor((255 * (im.detach() + 1) / 2).cpu().permute((0, 2, 3, 1)).numpy()[0],
                                  cv2.COLOR_RGB2BGR)
            im_output = cv2.cvtColor((255 * (output.detach() + 1) / 2).cpu().permute((0, 2, 3, 1)).numpy()[0],
                                  cv2.COLOR_RGB2BGR)
            cv2.imwrite('{}/input.jpeg'.format(config['train_params']['task_name']), im_input)
            cv2.imwrite('{}/output.jpeg'.format(config['train_params']['task_name']), im_output)
        
        loss = (crtierion(output, im) + config['train_params']['kl_weight']*kl)/(1+config['train_params']['kl_weight'])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        count += 1
        
    print('Finished epoch: {} | Loss : {:.4f} '.
          format(epoch_idx + 1,
                 np.mean(losses)))
    return np.mean(losses)