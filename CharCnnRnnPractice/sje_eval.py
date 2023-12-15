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
from char_cnn_rnn.eval_functions import encode_data, eval_classify, eval_retrieval

def main(args):
    rng_init(args.seed)