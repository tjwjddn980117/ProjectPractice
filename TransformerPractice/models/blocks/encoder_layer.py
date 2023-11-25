from torch import nn
from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionWiseFeedForward

class Encoder(nn.Module):
    def __init__(self, d_model, n_head, hidden, drop_prob):
        super(Encoder).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=hidden)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
    
    def forward(self, x):
        
        return x