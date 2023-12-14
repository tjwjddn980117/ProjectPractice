import os
import random
import argparse
from collections import OrderedDict

from tqdm import tqdm

import torch
import torchfile
from torch.utils.data import DataLoader

from utils import rng_init
import char_cnn_rnn as ccr

def encode_data(net_txt, net_img, data_dir, split, num_txts_eval, batch_size, device):
    '''
    Encoder for preprocessed Caltech-UCSD Birds 200-2011 and Oxford 102
    Category Flowers datasets, used in ``Learning Deep Representations of
    Fine-grained Visual Descriptions``.

    Warning: if you decide to not use all sentences (i.e., num_txts_eval > 0),
    sentences will be randomly sampled and their features will be averaged to
    provide a class representation. This means that the evaluation procedures
    should be performed multiple times (using different seeds) to account for
    this randomness.
    
    Arguments:
        net_txt (torch.nn.Module): text processing network.
        net_img (torch.nn.Module): image processing network.
        data_dir (string): path to directory containing dataset files.
        split (string): which data split to load.
        num_txts_eval (int): number of textual descriptions to use for each
            class (0 = use all). The embeddings are averaged per-class.
        batch_size (int): batch size to split data processing into chunks.
        device (torch.device): which device to do computation in.

    Returns:
        cls_feats_img (list of torch.Tensor): list containing precomputed image
            features for each image, separated by class.
        cls_feats_txt (torch.Tensor): tensor containing precomputed (and
            averaged) textual features for each class.
        cls_list (list of string): list of class names.
    '''

    # for example 'valclasses.txt', 'textclasses.txt'
    path_split_file = os.path.join(data_dir, split+'classes.txt')
    cls_list = [line.rstrip('\n') for line in open(path_split_file)]

    cls_feats_img = []
    cls_feats_txt = []
    for cls in cls_list:
        '''
        prepare image data
        '''
        data_img_path = os.path.join(data_dir, 'images', cls + '.t7')
        data_img = torch.Tensor(torchfile.load(data_img_path))
        # cub and flowers datasets have 10 image crops per instance
        # we use only the first crop per instance
        feats_img = data_img[:, :, 0].to(device)
        if net_img is not None: # if we had net_img
            with torch.no_grad(): # extracting feature image with net_image
                feats_img = net_img(feats_img)
        # one cls_feature_image for one class
        cls_feats_img.append(feats_img)


        '''
        prepare image data    
        '''
        data_txt_path = os.path.join(data_dir, 'text_c10', cls + '.t7')
        data_txt = torch.LongTensor(torchfile.load(data_txt_path))

        # size of data_txt was [num_of_instance, num_of_description, num_of_lenght]
        # select T texts from all instances to represent this class
        data_txt = data_txt.permute(0, 2, 1)
        # after permute, data_tax will [num_of_inst, num_of_len, num_of_descript]

        total_txts = data_txt.size(0) * data_txt.size(1)
        # 2d flatten
        data_txt = data_txt.contiguous().view(total_txts, -1)

        if num_txts_eval > 0:
            num_txts_eval = min(num_txts_eval, total_txts)
            id_txts = torch.randperm(data_txt.size(0))[:num_txts_eval]
            data_txt = data_txt[id_txts]