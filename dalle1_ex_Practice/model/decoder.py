import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder with couple of residual blocks followed by conv transpose relu layers. 
    """
    def __init__(self, embedding_dim):
        '''
        Decoder with couple of residual blocks followed by conv transpose relu layers. 

        Arguments:
            num_embeddings (int) : the number of embedding/channel.
            
        Parameters:
            _ (tensor) : [B, num_embeddings, X, X]. 
        
        Returns:
            _ (tensor) : [B, C, X\*8, X\*8]. 
        ''' 
        super(Decoder, self).__init__()
        
        self.decoder_layers = nn.ModuleList([
            # [B, 64, X, X] 
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            # [B, 64, X*2, X*2] 
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            # [B, 32, X*4, X*4] 
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            # [B, 16, X*8, X*8] 
            nn.Conv2d(16, 3, 1),
            # [B, 3, X*8, X*8] 
            nn.Tanh()
        ])
        
        self.residuals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU())
        ])
        
        self.decoder_quant_conv = nn.Conv2d(embedding_dim, 64, 1)
        
    
    def forward(self, x):
        out = self.decoder_quant_conv(x)
        for layer in self.residuals:
            out = layer(out)+out
        for idx, layer in enumerate(self.decoder_layers):
            out = layer(out)
        return out
        


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import yaml
    decoder = Decoder()
    
    out = decoder(torch.rand((3, 64, 14, 14)))
    print(out.shape)