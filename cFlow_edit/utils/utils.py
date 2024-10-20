import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pdb
import numpy as np
import os
import torch
from torchvision.utils import save_image


EPS = 1e-6

def ncc(a,v, zero_norm=True):

    a = a.flatten()
    v = v.flatten()

    if zero_norm:

        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)

    else:

        a = (a) / (np.std(a) * len(a))
        v = (v) / np.std(v)

    return np.correlate(a,v)

def variance_ncc_dist(sample_arr, gt_arr):

    def pixel_wise_xent(m_samp, m_gt, eps=1e-8):


        log_samples = np.log(m_samp + eps)

        return -1.0*np.sum(m_gt*log_samples, axis=-1)

    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y
    :return: 
    """
#    pdb.set_trace()
    gt_arr = gt_arr[0]
    sample_arr = sample_arr[0]
    mean_seg = np.mean(sample_arr, axis=0)

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    sX = sample_arr.shape[1]
    sY = sample_arr.shape[2]

    E_ss_arr = np.zeros((N,sX,sY))
    for i in range(N):
        E_ss_arr[i,...] = pixel_wise_xent(sample_arr[i,...], mean_seg)
        # print('pixel wise xent')
        # plt.imshow( E_ss_arr[i,...])
        # plt.show()
#    pdb.set_trace()

    E_ss = np.mean(E_ss_arr, axis=0)

    E_sy_arr = np.zeros((M,N, sX, sY))
    for j in range(M):
        for i in range(N):
            E_sy_arr[j,i, ...] = pixel_wise_xent(sample_arr[i,...], gt_arr[j,...])

    E_sy = np.mean(E_sy_arr, axis=1)

    ncc_list = []
#    pdb.set_trace()

    for j in range(M):

        ncc_list.append(ncc(E_ss, E_sy[j,...]))

    return (1/M)*sum(np.array(ncc_list))


def pdist(a,b):
    N = a.shape[1]
    M = b.shape[1]
    H = a.shape[-2]
    W = a.shape[-1]
#    C = a.shape[2]

    aRep = a.repeat(1,M,1,1).view(-1,N,M,H,W)
    bRep = b.repeat(1,N,1,1).view(-1,M,N,H,W).transpose(1,2)

    inter = (aRep & bRep).float().sum(-1).sum(-1) + EPS
    union = (aRep | bRep).float().sum(-1).sum(-1) + EPS
    IoU = inter/union
    dis = (1-IoU).mean(-1).mean(-1)
    return dis

def ged(seg,prd):
#    pdb.set_trace()
    seg = seg.type(torch.ByteTensor)
    prd = prd.type_as(seg)

    dSP = pdist(seg,prd)
    dSS = pdist(seg,seg)
    dPP = pdist(prd,prd)

    return (2*dSP - dSS - dPP)

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def save_mask_prediction_example(mask, pred, iter):
	plt.imshow(pred[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_prediction.png")
	plt.imshow(mask[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_mask.png")

def save_results(samples, paths, output_dir='/data/Result/'):
    """
    Save the sampled segmentation results as images and tensors.

    Args:
        samples (torch.Tensor): Sampled outputs of shape [B, 16, 2, H, W].
        paths (list of str): Original image paths, one per batch element.
        output_dir (str): Base directory where results will be saved.
    """
    # Iterate over each sample in the batch.
    batch_size, num_samples, _, _, _ = samples.shape
    for i in range(batch_size):
        # Extract the case name from the path.
        case_name = os.path.basename(os.path.dirname(paths[i]))

        # Create directories for saving images and tensors.
        image_save_dir = os.path.join(output_dir, 'image', case_name)
        tensor_save_dir = os.path.join(output_dir, 'tensor', case_name)
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(tensor_save_dir, exist_ok=True)

        # Iterate over each sampled output for this case.
        for j in range(num_samples):
            # Extract the j-th sampled output of shape [2, H, W].
            output_tensor = samples[i, j]

            # Save as a .pt tensor file.
            tensor_save_path = os.path.join(tensor_save_dir, f'output{j}.pt')
            torch.save(output_tensor, tensor_save_path)

            # Save the first channel (segmentation mask) as an image.
            image_save_path = os.path.join(image_save_dir, f'output{j}.png')
            # Normalize to [0, 1] if necessary and save as PNG.
            save_image(output_tensor[0], image_save_path)

            print(f'Saved tensor to {tensor_save_path} and image to {image_save_path}')
