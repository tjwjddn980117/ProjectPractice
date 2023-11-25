from torch import nn

class TokenEmbeddings(nn.Embedding):
    def __init__(self, vocal_size, d_model):
        # nn.Embedding
        # the token witch index == 1 is padding token
        super(TokenEmbeddings).__init__(vocal_size,d_model,padding_idx=1)