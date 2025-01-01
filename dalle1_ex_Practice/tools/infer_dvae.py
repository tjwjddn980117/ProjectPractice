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