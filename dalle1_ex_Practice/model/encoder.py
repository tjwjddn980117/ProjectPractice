import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder is conv relu blocks followed by couple of residual blocks. 
    Last 1x1 conv converts to logits with num_embeddings as output size. 
    """
    def __init__(self, num_embeddings):
        '''
        Encoder is conv relu blocks followed by couple of residual blocks. 
        Last 1x1 conv converts to logits with num_embeddings as output size. 

        Arguments:
            num_embeddings (int): the number of embedding/channel. 
            
        Inputs:
            [B, C, X, X]. 
        
        Outputs:
            [B, num_embeddings, X/8, X/8]. 
        '''
        super(Encoder, self).__init__()
        # Encoder is just Conv relu blocks
        self.encoder_layers = nn.ModuleList([
            # [B, C, X, X]
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            # [B, 32, X/2, X/2] 
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            # [B, 64, X/4, X/4]
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            # [B, 64, X/8, X/8] 
        ])
        self.residuals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding = 1),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU())
        ])
        self.encoder_quant_conv = nn.Sequential(
            nn.Conv2d(64, num_embeddings, 1))
        
    
    def forward(self, x):
        out = x
        for layer in self.encoder_layers:
            out = layer(out)
        for layer in self.residuals:
            out = out + layer(out)
        # 1x1 conv
        out = self.encoder_quant_conv(out)
        return out