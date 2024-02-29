import numpy as np
import torch

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from vqvae import VQVAE

torch.set_printoptions(linewidth=160)