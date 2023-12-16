import os
import argparse

from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torch
from dataset import MultimodalDataset
from torch.utils.data import DataLoader

from utils import rng_init, init_weights
import char_cnn_rnn.char_cnn_rnn as ccr

def sje_loss(feat1, feat2):
    ''' Structured Joint Embedding loss '''
    # similarity score matrix (rows: fixed feat2, columns: fixed feat1)
    # size of feat1 and feat2 is same as (B, 1024)
    # then, you can canculate the size (B, B) after matmul feat1 and feat2
    scores = torch.matmul(feat2, feat1.t()) # (B, B)

    # diagonal: matching pairs
    # pick the highest score in scores board
    ''' [[1], [2], [3]]'''
    diagonal = scores.diag().view(scores.size(0), 1) # (B, 1)
    
    # repeat diagonal scores on rows
    ''' [[1, 1, 1], [2, 2, 2], [3, 3, 3]]'''
    diagonal = diagonal.expand_as(scores) # (B, B)

    # calculate costs
    # Use the clamp to make the matrix free of negative numbers
    # When calculated in this manner, 
    # images similar to oneself all yield relatively high scores. 
    # The reduction in the average cost implies 
    #   that the latent space is becoming more refined and sophisticated.
    cost = (1 + scores - diagonal).clamp(min=0) # (B, B)

    # clear diagonals (matching pairs are not used in loss computation)
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
    writer = SummaryWriter(os.path.join(args.checkpoint_dir, args.save_file))

    # net_txt is the model
    net_txt = ccr.char_cnn_rnn(args.dataset, args.model_type).to(device)
    # you can init model's weights with apply function.
    # apply function works to define different initial methods with each layers
    net_txt.apply(init_weights)

    optim_txt = torch.optim.RMSprop(net_txt.parameters(), lr=args.learning_rate)
    sched_txt = torch.optim.lr_scheduler.ExponentialLR(optim_txt,args.learning_rate_decay)

    acc1_smooth = acc2_smooth = 0

    for epoch in tqdm(range(args.epochs), position=1):
        for i, data in enumerate(tqdm(loader, position=0)):
            iter_num = (epoch * loader_len) + i + 1 # function that studied datas

            net_txt.train()
            img = data['img'].to(device)
            txt = data['txt'].to(device)
            # get net_txt, feat_txt has been (B,1024)
            feat_txt = net_txt(txt)
            # The image data is already stored as (B, 1024), where B is the batch size. 
            # It utilizes data that has been pooled from the original images of size 224x224 through GoogLeNet,
            # resulting in 1,024 dimensions.
            feat_img = img

            # text to image loss
            loss1, acc1 = sje_loss(feat_txt, feat_img)
            loss2 = acc2 = 0
            if args.symmetric:
                loss2, acc2 = sje_loss(feat_img, feat_txt)
            loss = loss1 + loss2

            acc1_smooth = 0.99 * acc1_smooth + 0.01 * acc1
            acc2_smooth = 0.99 * acc2_smooth + 0.01 * acc2

            net_txt.zero_grad()
            loss.backward()
            optim_txt.step()

            # add_scalar function's parammeter is (tag, scalar, epoch)
            writer.add_scalar('train/loss1', loss1.item(), iter_num)
            writer.add_scalar('train/loss2', loss2.item(), iter_num)
            writer.add_scalar('train/acc1', acc1, iter_num)
            writer.add_scalar('train/acc2', acc2, iter_num)
            writer.add_scalar('train/acc1_smooth', acc1_smooth, iter_num)
            writer.add_scalar('train/acc2_smooth', acc2_smooth, iter_num)
            writer.add_scalar('train/lr', sched_txt.get_lr()[0], iter_num)

            # gramma of tqdm
            if (iter_num % args.print_event) == 0:
                run_info = (
                        'epoch: [{:3d}/{:3d}] | step: [{:4d}/{:4d}] | '
                        'loss: {:.4f} | loss1: {:.4f} | loss2: {:.4f} | '
                        'acc1: {:.2f} | acc2: {:.2f} | '
                        'acc1_smooth: {:.3f} | acc2_smooth: {:.3f} | '
                        'lr: {:.8f}'
                ).format(epoch+1, args.epochs, (i+1), loader_len,
                        loss, loss1, loss2,
                        acc1, acc2,
                        acc1_smooth, acc2_smooth,
                        sched_txt.get_lr()[0])
                tqdm.write(run_info)
            
        net_txt.eval()
        tqdm.write('Saving checkpoint to: {}'.format(ckpt_path))
        # we save our net evaluation in 'ckpt_path'
        torch.save(net_txt.state_dict(), ckpt_path)

        sched_txt.step()

    writer.close()

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
    parser.add_argument('--train_split', type=str, required=True,
            choices=['train', 'val', 'test', 'trainval', 'all'],
            help='Which dataset split is used for training')

    parser.add_argument('--epochs', type=int,
            default=300,
            help='Number of epochs')
    parser.add_argument('--batch_size', type=int,
            default=40,
            help='Training batch size')
    parser.add_argument('--symmetric', type=bool,
            default=True,
            help='Whether or not to use symmetric form of SJE')
    parser.add_argument('--learning_rate', type=float,
            default=0.0004,
            help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float,
            default=0.98,
            help='Learning rate decay')

    parser.add_argument('--seed', type=int, required=True,
            help='RNG seed')
    parser.add_argument('--use_gpu', type=bool,
            default=True,
            help='Whether or not to use GPU')
    parser.add_argument('--print_every', type=int,
            default=100,
            help='How many steps/mini-batches between printing out the loss')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
            help='Output directory where model checkpoints get saved at')
    parser.add_argument('--save_file', type=str, required=True,
            help='Name to autosave model checkpoint to')

    args = parser.parse_args()
    main(args)