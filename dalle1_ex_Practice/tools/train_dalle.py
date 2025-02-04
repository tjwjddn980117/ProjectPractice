import yaml
import argparse
import torch
import random
import os
import numpy as np
from tqdm import tqdm
from model.discrete_vae import DiscreteVAE
from model.dalle import DallE
from torch.utils.data.dataloader import DataLoader
from datasets.mnist_color_texture_dataset import MnistVisualLanguageDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt_counts = 0

def train_for_one_epoch(epoch_idx, model, loader, optimizer, config):
    """
    Method to run the training for one epoch.

    Inputs:
        epoch_idx (int): iteration number of current epoch. 
        model (nn.Module): Dalle model. 
        mnist_loader (DataLoader): Data loder. 
        optimizer (optimizer): optimzier to be used taken from config. 
        crtierion (crtierion): For computing the loss. 
        config (dict): configuration for the current run. 

    Outputs:
        losses (float): the mean of losses. each losses are from each batch(text and image). 
    """
    losses = []
    for data in tqdm(loader):
        im = data['image'] 
        text_tokens = data['text_tokens']
        im = im.float().to(device)
        text = text_tokens.long().to(device)
        optimizer.zero_grad()
        
        _, loss_text, loss_image = model(im, text)
        loss = (loss_text*1 + loss_image*config['train_params']['dalle_image_loss']) / (1+config['train_params']['dalle_image_loss'])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print('Finished epoch: {} | Modelling Loss : {:.4f} '.
          format(epoch_idx + 1,
                 np.mean(losses)))
    return np.mean(losses)

