import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.eopch_timer import epoch_time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1: # this show weight's dim is not a scalar, only over verctor
        nn.init.kaiming_normal(m.weight.data) # init weights for kaiming_normal

model = Transformer(src_pad_idx=src_pad_idx,
                    tar_pad_idx=tar_pad_idx,
                    tar_sos_idx=tar_sos_idx,
                    enc_vocal_size=enc_voc_size,
                    dec_vocal_size=dec_voc_size,
                    max_len=max_len,
                    d_model=d_model,
                    hidden=hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)

optimizer = Adam(params=model.parameters(),
                 lr = init_lr, weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True,
                                                 factor=factor, patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous(-1, output.shape[-1])
        tar = trg[:, 1:].contiguous().view(-1)