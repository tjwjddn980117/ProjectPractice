import os
import random
import argparse
from collections import OrderedDict

from tqdm import tqdm

import torch

from utils import rng_init
import char_cnn_rnn.char_cnn_rnn as ccr
from char_cnn_rnn.eval_functions import encode_data, eval_classify, eval_retrieval

def main(args):
    rng_init(args.seed)
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    net_txt = ccr.char_cnn_rnn(args.dataset, args.model_type)
    net_txt.load_state_dict(torch.load(args.model_path, map_location=device))
    net_txt = net_txt.to(device)
    net_txt.eval()

    cls_feats_img, cls_feats_txt, cls_list = encode_data.encode_data(net_txt, None, args.data_dir,
            args.eval_split, args.num_txts_eval, args.batch_size, device)
    
    mean_ap, cls_stats = eval_retrieval.eval_retrieval(cls_feats_img, cls_feats_txt, cls_list)

    print('----- RETRIEVAL -----')
    if args.print_class_stats:
        print('  PER CLASS:')
        for name, stats in cls_stats.items():
            print(name)
            for k, ap in stats.items():
                print('{:.4f}: AP@{}'.format(ap, k))
        print()

    print('  mAP:')
    for k, v in mean_ap.items():
        print('{:.4f}: mAP@{}'.format(v, k))
    print('---------------------')
    print()

    avg_acc, cls_stats = eval_classify(cls_feats_img, cls_feats_txt, cls_list)
    print('--- CLASSIFICATION --')
    if args.print_class_stats:
        print('  PER CLASS:')
        for name, stats in cls_stats.items():
            print('{:.4f}: {}'.format(stats['correct'] / stats['total'], name))
        print()

    print('Average top-1 accuracy: {:.4f}'.format(avg_acc))
    print('---------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
            choices=['birds', 'flowers'],
            help='Dataset type')
    parser.add_argument('--model_type', type=str, required=True,
            choices=['cvpr', 'icml'],
            help='Model type')
    parser.add_argument('--data_dir', type=str, required=True,
            help='Data directory')
    parser.add_argument('--eval_split', type=str, required=True,
            choices=['train', 'val', 'test', 'trainval', 'all'],
            help='Which dataset split to use')

    parser.add_argument('--model_path', type=str, required=True,
            help='Model checkpoint path')
    parser.add_argument('--num_txts_eval', type=int,
            default=0,
            help='Number of texts to use per class (0 = use all)')
    parser.add_argument('--print_class_stats', type=bool,
            default=True,
            help='Whether to print per class statistics or not')
    parser.add_argument('--batch_size', type=int,
            default=40,
            help='Evaluation batch size')

    parser.add_argument('--seed', type=int, required=True,
            help='Which RNG seed to use')
    parser.add_argument('--use_gpu', type=bool,
            default=True,
            help='Whether or not to use GPU')

    args = parser.parse_args()
    main(args)