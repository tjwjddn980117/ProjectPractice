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
    