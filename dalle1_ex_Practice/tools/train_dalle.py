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

