import os
import torch
import torchfile
from torch.utils.data import Dataset

from char_cnn_rnn.char_cnn_rnn import labelvec_to_onehot

class MultimodalDataset(Dataset):
    '''
    Preprocessed Caltech-UCSD Birds 200-2011 and Oxford 102 Category Flowers
    datasets, used in ``Learning Deep Representations of Fine-grained Visual
    Descriptions``.

    Download data from: https://github.com/reedscot/cvpr2016.

    Arguments:
        data_dir (string): path to directory containing dataset files.
        split (string): which data split to load.
    '''
    def __init__(self, data_dir, split):
        super(MultimodalDataset).__init__()
        possible_splits = ['train', 'val', 'test', 'trainval', 'all']
        assert split in possible_splits, 'Split should be: {}'.format(', '.join(possible_splits))
        