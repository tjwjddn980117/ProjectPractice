import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):
    def __init__(self, dec_vocal_size, max_len, d_model, hidden, n_head, n_layers, drop_prob, device):
        super(Decoder).__init__()
        self.embedding = TransformerEmbedding(d_model=d_model, vocal_size=dec_vocal_size, max_len=max_len, device=device, drop_prob=drop_prob)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, n_head=n_head, hidden=hidden, drop_prob=drop_prob)
                                    for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, dec_vocal_size)

    def forward(self, tar, enc_src, tar_mask, src_mask):
        tar = self.embedding(tar)
        for layer in self.layers:
            tar = layer(tar, enc_src, tar_mask, src_mask)
        
        # pass the linear
        tar = self.linear(tar)
        return tar