from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
# it is dictionary, but it can easy to use. (use '.')
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'birds'
__C.EMBEDDING_TYPE = 'cnn-rnn'
__C.CONFIG_NAME = ''
__C.DATA_DIR = ''

__C.GPU_ID = '0'
__C.CUDA = True

__C.WORKERS = 6

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 64

# Test options
__C.TEST = edict()
__C.TEST.B_EXAMPLE = True
__C.TEST.SAMPLE_NUM = 30000

# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.VIS_COUNT = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.FLAG = True
__C.TRAIN.NET_G = ''
__C.TRAIN.NET_D = ''

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 2.0
__C.TRAIN.COEFF.UNCOND_LOSS = 0.0
__C.TRAIN.COEFF.COLOR_LOSS = 0.0

# Modal options
__C.GAN = edict()
__C.GAN.EMBEDDING_DIM = 128
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 64
__C.GAN.Z_DIM = 100
__C.GAN.NETWORK_TYPE = 'default'
__C.GAN.R_NUM = 2
__C.GAN.B_CONDITION = False

__C.TEXT = edict()
__C.TEXT.DIMENSION = 1024

def _merge_a_into_b(a, b):
    '''
    Merge config dictionary a into config dictionary b, 
    clobbering the options in b whenever they are also specified in a.
    Arguments:
        a (edict): the edict which be new edict
        b (edict): the edict which was the origin
    '''

    if type(a) is not edict:
        return

    for key, value in a.items():
        # a must specify keys that are in b
        if key not in b:
            raise KeyError('{} is not a valid config key'.format(key))
        
        # the types must match, too
        old_type = type(b[key])
        if old_type is not type(value):
            # if a's value isn't Numpy, we should change value to Numpy
            if isinstance(b[key], np.ndarray):
                value = np.array(value, dtype=b[key].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[key]),type(value), key))
        
        # recursively merge dicts
        if type(key) is edict:
            try:
                _merge_a_into_b(a[key], b[key])
            except:
                print('Error under config key: {}'.format(key))
                raise
        else:
            # it just repace the value b to a
            b[key] = value

def cfg_from_file(filename):
    '''
    Load a config file and merge it into the default options.
    '''
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)