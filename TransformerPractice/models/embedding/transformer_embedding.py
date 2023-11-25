from torch import nn

from models.embedding.token_embeddings import TokenEmbeddings
from models.embedding.positional_encoding import PositionalEncoding

class TransformerEmbedding(nn.Module):

    def __init__(self, d_model, vocal_size, max_len, device, drop_prob):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbeddings(vocal_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)
    
    def forward(self, x):
        # size of x is [batch_size, seq_len]
        tok_emb = self.tok_emb(x) # size of tok_emb is [batch_size, seq_len, d_model]
        pos_emb = self.pos_emb(x) # size of pos_emb is [seq_len, d_model] 
        # it could copy because it's just imformation of position, so the pos_emb is fixed.
        # that's the reason why it can be copied
        return self.drop_out(tok_emb+pos_emb) # size of return is [batch_size, seq_len, d_model] with broadcast