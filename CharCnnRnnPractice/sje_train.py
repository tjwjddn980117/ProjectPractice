import os
import argparse

from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torch
from dataset import MultimodalDataset
from torch.utils.data import DataLoader

from utils import rng_init, init_weights
import char_cnn_rnn as ccr

def sje_loss(feat1, feat2):
    ''' Structured Joint Embedding loss '''
    # similarity score matrix (rows: fixed feat2, columns: fixed feat1)
    scores = torch.matmul(feat2, feat1.t()) # (B, B)

    # diagonal: matching pairs
    ''' [[1], [2], [3]]'''
    diagonal = scores.diag().view(scores.size(0), 1) # (B, 1)
    
    # repeat diagonal scores on rows
    ''' [[1, 1, 1], [2, 2, 2], [3, 3, 3]]'''
    diagonal = diagonal.expand_as(scores) # (B, B)

    # calculate costs
    cost = (1 + scores - diagonal).clamp(min=0) # (B, B)

    # clear diagonals (matching pairs are not used in loss computatioin)
    # cost[torch.eye(cost.size(0)).bool()] = 0 # (B, B) for torch==1.2.0
    cost[torch.eye(cost.size(0), dtype=torch.uint8)] = 0 # (B, B)

    # sum and average costs
    denom = cost.size(0) * cost.size(1)
    loss = cost.sum() / denom


    # batch accuracy
    max_ids = torch.argmax(scores, dim=1)
    ground_truths = torch.LongTensor(range(scores.size(0))).to(feat1.device)
    num_correct = (max_ids == ground_truths).sum().float()
    accuracy = 100 * num_correct / cost.size(0)

    return loss, accuracy