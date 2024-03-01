import numpy as np
import torch

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from vqvae import VQVAE

torch.set_printoptions(linewidth=160)
 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def save_img_tensors_as_grid(img_tensors, nrows, f):
    '''
    The function with save image as grid.

    Inputs:
        img_tensors(tensor): [B,C,H,W]
        nrows(int): number of rows
        f(str): name of saving file.
    '''
    img_tensors = img_tensors.permute(0, 2, 3, 1)
    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array[imgs_array < -0.5] = -0.5
    imgs_array[imgs_array > 0.5] = 0.5
    imgs_array = 255 * (imgs_array + 0.5)
    (batch_size, img_size) = img_tensors.shape[:2]
    ncols = batch_size // nrows
    img_arr = np.zeros((nrows * batch_size, ncols * batch_size, 3))
    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_size
        row_end = row_start + img_size
        col_start = col_idx * img_size
        col_end = col_start + img_size
        img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]

    Image.fromarray(img_arr.astype(np.uint8), "RGB").save(f"{f}.jpg")

use_ema = True
model_args = {
    "in_channels": 3,
    "num_hiddens": 128,
    "num_downsampling_layers": 2,
    "num_residual_layers": 2,
    "num_residual_hiddens": 32,
    "embedding_dim": 64,
    "num_embeddings": 512,
    "use_ema": use_ema,
    "decay": 0.99,
    "epsilon": 1e-5,
}
model = VQVAE(**model_args).to(device)

batch_size = 32
workers = 10
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)

data_root = "../data"
train_dataset = CIFAR10(data_root, True, transform, download=True)
train_data_variance = np.var(train_dataset.data / 255)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
)

beta = 0.25

# Initialize optimizer.
train_params = [params for params in model.parameters()]
lr = 3e-4   
optimizer = optim.Adam(train_params, lr=lr)
criterion = nn.MSELoss()

# Train model.
epochs = 7
eval_every = 100
best_train_loss = float("inf")
model.train()

for epoch in range(epochs):
    total_train_loss = 0
    total_recon_error = 0
    n_train = 0