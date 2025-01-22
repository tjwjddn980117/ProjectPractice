import shutil
import yaml
import argparse
import torch
import os
import torchvision
from model.discrete_vae import DiscreteVAE
from datasets.mnist_color_texture_dataset import MnistVisualLanguageDataset
from torchvision.utils import make_grid
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inference(args):
    r"""
    Method to infer discrete vae and get
    reconstructions
    :param args:
    :return:
    """
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    model = DiscreteVAE(
        num_embeddings=config['model_params']['vae_num_embeddings'],
        embedding_dim=config['model_params']['vae_embedding_dim']
    )
    model.to(device)
    if os.path.exists('{}/{}'.format(config['train_params']['task_name'],
                                     config['train_params']['vae_ckpt_name'])):
        print('Found checkpoint... Inferring from that')
        model.load_state_dict(torch.load('{}/{}'.format(config['train_params']['task_name'],
                                                        config['train_params']['vae_ckpt_name']), map_location=device))
    else:
        print('No checkpoint found at {}/{}... Exiting'.format(config['train_params']['task_name'],
                                     config['train_params']['vae_ckpt_name']))
        return
    model.eval()
    mnist = MnistVisualLanguageDataset('test', config['dataset_params'])
    