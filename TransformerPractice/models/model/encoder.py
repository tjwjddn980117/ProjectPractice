import torch
from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self, enc_vocal_size, max_len, d_model, hidden, n_head, n_layers, drop_prob, device):
        self.embedding = TransformerEmbedding(d_model=d_model, vocal_size=enc_vocal_size, max_len=max_len, drop_prob=drop_prob, device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, n_head=n_head, hidden=hidden, drop_prob=drop_prob)
                                     for _ in range(n_layers)])
    
    def forward(self, x, src_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x