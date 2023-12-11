from torch import nn
from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, hidden, drop_prob):
        super(DecoderLayer,self).__init__()
        self.attention1 = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.attention2 = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, tar_mask, src_mask):

        # 1. self attention 
        _x = dec
        x = self.attention1(q=dec, k=dec, v=dec, mask=tar_mask)

        # 2. normalize
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. decoder attenttion
            _x = x
            x = self.attention2(q=x, k=enc, v=enc, mask=src_mask)

            # 4. normalize
            x = self.dropout2(x)
            x = self.norm2(x + _x)
        
        # 5. ffn
        _x = x
        x = self.ffn(x)
        
        # 6. normalize
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x