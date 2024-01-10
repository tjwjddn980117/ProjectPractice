import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torchvision.models as models

import random
from tqdm.auto import tqdm
import os
mydir = os.getcwd()

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore')

def seed_everything(seed):
    '''
    define seed with fixed value.
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed=42
seed_everything(seed) # Seed 고정

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

data_path = mydir + '\content'
train_df = pd.read_csv(data_path+'\\train.csv')
test_df = pd.read_csv(data_path+'\\test.csv')

