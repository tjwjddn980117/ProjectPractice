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

def eval_retrieval(cls_feats_img, cls_feats_txt, cls_list, k_values=[1,5,10,50]):
    '''
    Retrieval evaluation (Average Precision).

    Arguments:
        cls_feats_img (list of torch.Tensor): list containing precomputed image
            features for each image, separated by class.
        cls_feats_txt (torch.Tensor): tensor containing precomputed (and
            averaged) textual features for each class.
        cls_list (list of string): list of class names.
        k_values (list, optional): list of k-values to use for evaluation.

    Returns:
        map_at_k (OrderedDict): dictionary whose keys are the k_values and the
            values are the mean Average Precision (mAP) for all classes.
        cls_stats (OrderedDict): dictionary whose keys are class names and each
            entry is a dictionary whose keys are the k_values and the values
            are the Average Precision (AP) per class.
    '''
    total_num_cls = cls_feats_txt.size(0)
    total_num_img = 
    return 