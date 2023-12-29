import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo

from utils import *

# ############## G networks ################################################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

# Upsale the spatial size by a factor of 2