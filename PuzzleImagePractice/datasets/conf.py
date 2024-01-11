import os
import numpy as np
import pandas as pd
import torch
import random

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

mydir = os.getcwd()

data_path = mydir + '\\content'
data_path = 'C:\\Users\\qowor\\Desktop\\open'
save_origin_path = data_path+'\\origin'
save_augment_path = data_path+'\\augment'
train_df = pd.read_csv(data_path+'\\train.csv')
test_df = pd.read_csv(data_path+'\\test.csv')