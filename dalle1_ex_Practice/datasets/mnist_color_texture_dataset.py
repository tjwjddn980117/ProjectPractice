import numpy as np
import cv2
import os
import random
import matplotlib.colors as mcolors
import torch
import json
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


def get_square_crop(image):
    '''
    crop to square size. 
    Cut the longer side, either horizontal or vertical, to match the length of the shorter side. 
    When cutting, cut off both ends of the long side based on the center.

    Inputs:
        image (tensor): [H, W, C]
    
    Outputs:
        image (tensor): [X, X, C]
    '''
    h,w = image.shape[:2]
    if h > w:
        return image[(h - w)//2:-(h - w)//2, :, :]
    else:
        return image[:, (w - h) // 2:-(w - h) // 2, :]