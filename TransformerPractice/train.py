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
        output = model(src, trg[:, :-1]) # this is the reason why the last sequence should prediced
        # output size will be [batch_size, seq_len, dec_vocal_size]
        # output_reshape size will be [batch_size*seq_len, dec_vocal_size]
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        # tar size was [batch_size, seq_len],
        # but after resize, there size will be [batch_size*seq_len]
        trg = trg[:, 1:].contiguous().view(-1) # this is the reason why the first sequence is '<sos>'

        loss = criterion(output_reshape, trg)
        loss.backward()
        # When gradient exceeds a certain threshold, clipping is done.
        torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        optimizer.step()
        
        # use 'loss.item()' is more safe
        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
    
    return epoch_loss / len(iterator)

def evaluation(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            # output size will be [batch_size, seq_len, dec_vocal_size]
            # output_reshape size will be [batch_size*seq_len, dec_vocal_size]
            output = model(src, trg[:,:-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            # under this code is evaluation method
            total_bleu = []
            for j in range(batch_size):
                try: 
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    # Tensor containing the index of the most likely words at each position predicted by the model
                    # (dim=1) means about picking the biggest numbers in the direction of the row,
                    # and [1] means about the index of the row
                    # this 'output_words' tensor's size will be [seq_len]
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    # 1. if batch_size is bigger than trg[j]
                    # 2. can't find index in loader.target.vocab
                    pass
            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu

def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluation(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)