import yaml
import argparse
import torch
import random
import os
import torchvision
import numpy as np
from einops import rearrange
from tqdm import tqdm
from model.discrete_vae import DiscreteVAE
from model.dalle import DallE
from datasets.mnist_color_texture_dataset import MnistVisualLanguageDataset
from torchvision.utils import make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer(args):
    ######## Read the config file #######
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    ######## Set the desired seed value #######
    # Ignoring the fixed seed value
    seed = np.random.randint(0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    if not os.path.exists(config['train_params']['task_name']):
        os.mkdir(config['train_params']['task_name'])
    
    # Create db to fetch the configuration values like vocab size (should do something better)
    mnist = MnistVisualLanguageDataset('train', config['dataset_params'])
    
    ###### Load Discrete VAE#####
    
    vae = DiscreteVAE(
        num_embeddings=config['model_params']['vae_num_embeddings'],
        embedding_dim=config['model_params']['vae_embedding_dim']
    )
    vae.to(device)
    if os.path.exists('{}/{}'.format(config['train_params']['task_name'],
                                     config['train_params']['vae_ckpt_name'])):
        print('Found checkpoint... Taking vae from that')
        vae.load_state_dict(torch.load('{}/{}'.format(config['train_params']['task_name'],
                                                      config['train_params']['vae_ckpt_name']), map_location=device))
    else:
        print('No checkpoint found at {}/{}... Exiting'.format(config['train_params']['task_name'],
                                                               config['train_params']['vae_ckpt_name']))
        print('Train vae first')
        return
    vae.eval()
    vae.requires_grad_(False)