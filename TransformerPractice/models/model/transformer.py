import torch
from torch import nn

from models.model.encoder import Encoder
from models.model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, tar_pad_idx, tar_sos_idx, enc_vocal_size, dec_vocal_size, max_len, 
                 d_model, hidden, n_head, n_layers, drop_prob, device):
        super(Transformer).__init__()
        self.src_pad_idx = src_pad_idx
        self.tar_pad_idx = tar_pad_idx
        self.tar_sos_idx = tar_sos_idx
        self.device = device
        self.encoder = Encoder(enc_vocal_size=enc_vocal_size, max_len=max_len, 
                               d_model=d_model, hidden=hidden, n_head=n_head, 
                               n_layers=n_layers, drop_prob=drop_prob, 
                               device=device)
        
        self.decoder = Decoder(dec_vocla_size=dec_vocal_size, max_len=max_len, 
                               d_model=d_model, hidden=hidden, n_head=n_head,
                               n_layers=n_layers, drop_prob=drop_prob,
                               device=device)
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # if src size is [batch_size, sequence_length]
        # output will be [batch_size, 1, 1, sequence_lenght] with True / False
        return src_mask
    
    def make_tar_mask(self, tar):
        # size of tar is [batch_size, seq_length]
        # size of tar_pad_mask is [batch_size, 1, 1, seq_length]
        # size of tar_sub_mask is [seq_lenght, seq_length]
        # size of tar_mask is [batch_size, 1, seq_length, seq_length]
        tar_pad_mask = (tar != self.tar_pad_idx).unsqueeze(1).unsqueeze(3)
        tar_len = tar.shape[1]
        # tril == low triangular matrix / ByteTensor == type to Boolean
        tar_sub_mask = torch.tril(torch.ones(tar_len, tar_len)).type(torch.ByteTensor).to(self.device)
        # tar_pad_mask & tar_sub_mask can canculate because of broadcast
        # both tar_pad_mask and tar_sub_mask size be [batch_size, 1, seq_length, seq_length]
        tar_mask = tar_pad_mask & tar_sub_mask
        return tar_mask

    def forward(self, src, tar):
        src_mask = self.make_src_mask(src)
        tar_mask = self.make_tar_mask(tar)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(tar,enc_src, tar_mask, src_mask)
        return output
        