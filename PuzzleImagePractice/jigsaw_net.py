import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks.blocks import ConvBlock, DeconvBlock

class JIGSAW_NET(nn.Module):
    def __init__(self, in_channels):
        '''
        This is the basic jigsaw_net. We had batch-norm. This Block don't change the size.

        Arguments:
            in_channels (int): the number of channel.
        
        Inputs:
            x (nparray): [batch_size, channel, H, W].
        '''
        super(JIGSAW_NET, self).__init__()
        self.in_channels = in_channels

        # Contraction path
        self.conv1 = ConvBlock(in_channels, 16, 'he_normal')
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = ConvBlock(16, 32, 'he_normal')
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ConvBlock(32, 64, 'he_normal')
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = ConvBlock(64, 128, 'he_normal')
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = ConvBlock(128, 256, 'he_normal')

        # Expansion path
        self.convUp1 = DeconvBlock(256, 128, 'he_normal')
        self.convUp2 = DeconvBlock(128, 64, 'he_normal')
        self.convUp3 = DeconvBlock(64, 32, 'he_normal')
        self.convUp4 = DeconvBlock(32, 16, 'he_normal')


    def forward(self, x):
        x = x(self.conv1)