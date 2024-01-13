import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, init='he_normal'):
        '''
        This is the basic Convolution block. We had batch-norm. This Block don't change the size.

        Arguments:
            in_channels (int): the number of input channels.
            out_channels (int): the number of output channels.
            init (bool|str): the type of initializing weights of model.

        Inputs:
            x (nparray): [batch_size, channel, H, W].
        
        Outputs:
            x (nparray): [batch_size, channel, H, W].
        '''
        super(ConvBlock, self).__init__()      
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        if init == 'he_normal':
            nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        return x

class DeconvBlock(nn.Module):
    '''
        This is the basic Convolution block. We had batchnorm.

        Arguments:
            in_channels (int): the number of input channels.
            out_channels (int): the number of output channels.
            init (bool|str): the type of initializing weights of model.

        Inputs:
            x (nparray): [batch_size, channel, H, W].
            residual (nparray): [batch_size, channel', 2*H, 2*W].
        
        Outputs:
            x (nparray): [batch_size, channel, 2*H, 2*W].
    '''
    def __init__(self, in_channels, out_channels, init='he_normal'):
        super(DeconvBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.convBlock = ConvBlock(out_channels, out_channels, init='he_normal')

        if init == 'he_normal':
            nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x, residual):
        x = self.upsample(x)
        x = self.conv(x)
        x = F.relu(x)
        x = torch.cat((x,residual), dim=1)
        x = self.convBlock(x)
        return x
        