from config import CFG
import random, os
import numpy as np
import torch

def seed_everything(seed):
    '''
    This function is intended to control randomness to obtain consistent results. 
    ''' 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# seed_everything(CFG['SEED'])