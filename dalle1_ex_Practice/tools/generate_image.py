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