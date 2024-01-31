import time

from torch import nn, optim
from torch.optim import Adam

from model.JigsawNet import JIGSAW_NET
from utils.conf import *
from utils.epoch_time import epoch_time
from datasets.dataset import TRAIN_DATA_LOADER, TEST_DATA_LOADER

model = JIGSAW_NET(3)
model = model.to(device)

optimizer = Adam(params=model.parameters(), lr = INIT_LR, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True,
                                                 factor=FACTOR, patience=PATIENCE)

criterion = nn.CrossEntropyLoss()

def train(model, datasets, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for i, (batch, label) in enumerate(datasets):
        batch = batch.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(batch)
        output_reshape = output.contiguous().view(-1, output.shape[-1])

        loss = criterion(output_reshape, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(datasets)) * 100, 2), '% , loss :', loss.item())
    
    return epoch_loss / len(datasets)

def evaluation(model, datasets, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (batch, label) in enumerate(datasets):
            batch = batch.to(device)
            label = label.to(device)
            output = model(batch)
            output_reshape = output.contiguous().view(-1, output.shape[-1])

            loss = criterion(output_reshape, label)

            epoch_loss += loss.item()
        
    return epoch_loss / len(datasets)

def run(total_epoch, best_loss):
    train_losses, test_losses = [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, TRAIN_DATA_LOADER, optimizer, criterion)
        valid_loss = evaluation(model, TEST_DATA_LOADER, criterion)
        end_time = time.time()

        if step > WARMUP:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{:.4f}.pt'.format(valid_loss))

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tVal Loss: {valid_loss:.3f}')

    with open('result/train_loss.txt', 'w') as f:
        f.write(str(train_losses))
        
    with open('result/test_loss.txt', 'w') as f:
        f.write(str(test_losses))
if __name__ == '__main__':
    run(total_epoch=EPOCH, best_loss=INF)