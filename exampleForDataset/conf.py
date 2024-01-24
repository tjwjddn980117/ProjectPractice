import os
import numpy as np
import pandas as pd
import torch

# define device.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# parameter about dataset
mydir = os.getcwd()
#DATA_PATH = mydir + '\\content'
DATA_PATH = 'C:\\Users\\qowor\\Desktop\\open'
SAVE_ORIGIN_PATH = os.path.join(DATA_PATH, 'origin')
SAVE_AGUMENT_PATH = os.path.join(DATA_PATH, 'augment')
TRAIN_CSV = os.path.join(DATA_PATH, 'train.csv')
TEST_CSV = os.path.join(DATA_PATH, 'test.csv')

# parameter about model.
BATCH_SIZE = 32

# optimizer parameter setting
INIT_LR = 1e-5
FACTOR = 0.9
ADAM_EPS = 5e-9
PATIENCE = 10
WARMUP = 100
EPOCH = 1000
CLIP = 1.0
WEIGHT_DECAY = 5e-4
INF = float('inf')