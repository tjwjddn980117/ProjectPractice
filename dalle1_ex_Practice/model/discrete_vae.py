import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.quantizer import Quantizer

class DiscreteVAE(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=512):
        '''
        The class of DiscreteVAE. 

        Arguments:
            num_embeddings (int): the number of embeddings. 
            embedding_dim (int): the number of dimension of embedding. 
        
        Inputs:
            x (tensor): [B, C, H, W]. 
        
        Outputs:
            out (tensor): [B, C, H, W]. 
            kl (float): the difference between log_uniform and log_qy (KLD(P||Q)). 
            log_qy (tensor): [B, Pixel(HW), C]. log_softmax with logits. 
        '''
        super(DiscreteVAE, self).__init__()
        self.encoder = Encoder(num_embeddings=num_embeddings)
        self.quantizer = Quantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.decoder = Decoder(embedding_dim=embedding_dim)
    
    def get_codebook_indices(self, x):
        '''
        the function get codebook indices. 

        Arguments:
            x (tensor): [B, C, H, W]. 

        Outputs: 
            indices (tensor): [B, H/8, W/8]. 
        '''
        # x.shape = B,C,H,W
        enc_logits = self.encoder(x)
        # enc_logits.shape = [B, C, H/8, W/8]
        indices = torch.argmax(enc_logits, dim=1)
        # indices.shape = [B,H/8,W/8]
        return indices
    
    def decode_from_codebook_indices(self, indices):
        '''
        the function to decode from latent space. 

        Arguments:
            indices (tensor): [B, C, H, W]. 
        
        Outputs:
            decoder (tensor): [B, C, X*8, X*8]. 
        '''
        quantized_indices = self.quantizer.quantize_indices(indices)
        return self.decoder(quantized_indices)
        
    def forward(self, x):
        # x = [B, C, H, W]. 
        enc = self.encoder(x)
        # enc = [B, C, H/8, W/8]. 
        quant_output, kl, logits, log_qy = self.quantizer(enc)
        # qant_output = [B, D, H/8, W/8]. 
        # kl = float. 
        # log_qy = [B, Pixel, C]. 
        out = self.decoder(quant_output)
        # out = [B, C, H, W]. 
        return out, kl, log_qy