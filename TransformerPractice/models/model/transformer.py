import torch
from torch import nn

from models.model.encoder import Encoder
from models.model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_enc_idx, src_dec_idx, sos_dec_idx, enc_vocal_size, dec_vocal_size, max_len, 
                 d_model, hidden, n_head, n_layers, drop_prob, device):
        super(Transformer).__init__()
        self.src_enc_idx = src_enc_idx
        self.src_dec_idx = src_dec_idx
        self.sos_dec_idx = sos_dec_idx
        self.device = device
        self.encoder = Encoder(enc_vocal_size=enc_vocal_size, max_len=max_len, 
                               d_model=d_model, hidden=hidden, n_head=n_head, 
                               n_layers=n_layers, drop_prob=drop_prob, 
                               device=device)
        
        self.decoder = Decoder(dec_vocla_size=dec_vocal_size, max_len=max_len, 
                               d_model=d_model, hidden=hidden, n_head=n_head,
                               n_layers=n_layers, drop_prob=drop_prob,
                               device=device)
    
    
    def forward(self, src, tar):
        src_mask = self.make_src_mask(src)
        tar_mask = self.make_tar_mask(src)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(tar,enc_src, tar_mask, src_mask)
        return output
        