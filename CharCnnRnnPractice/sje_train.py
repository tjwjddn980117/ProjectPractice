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
    # Use the clamp to make the matrix free of negative numbers
    cost = (1 + scores - diagonal).clamp(min=0) # (B, B)

    # clear diagonals (matching pairs are not used in loss computatioin)
    # cost[torch.eye(cost.size(0)).bool()] = 0 # (B, B) for torch==1.2.0
    # make cost's diagonals to 0
    # it couldn't canculate their own similarity
    cost[torch.eye(cost.size(0), dtype=torch.uint8)] = 0 # (B, B)

    # sum and average costs
    # demon is the total number of sections in each matrix.
    # and evaluation of each demon is loss
    denom = cost.size(0) * cost.size(1)
    loss = cost.sum() / denom


    # batch accuracy
    # argmax is the function that return index (maximum number)
    max_ids = torch.argmax(scores, dim=1)
    # As ground truth, 
    # it was assumed that the index that is most similar to oneself is oneself.
    ground_truths = torch.LongTensor(range(scores.size(0))).to(feat1.device)
    num_correct = (max_ids == ground_truths).sum().float()
    accuracy = 100 * num_correct / cost.size(0)

    return loss, accuracy

def main(args):
    rng_init(args.seed)
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    dataset = MultimodalDataset(args.data_dir, args.train_split)
    loader = DataLoader(dataset, batch_size=args.batchsize)
    loader_len = len(loader)

    os.makedirs(os.path.join(args.checkpoint_dir, args.save_file), exist_ok=True)
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_name = '{}_{:.5f}_{}_{}_{}.pth'.format(
        args.save_file, args.learning_rate, args.symmetric, args.train_split, timestamp)
    ckpt_path = os.path.join(args.checkpoint_dir, args.save_file, model_name)
    