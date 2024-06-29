from torch import nn
from conf import *
from Layers import *

class Transformer(nn.Module):
    '''
    Transformer block. 
    '''
    def __init__(self, src_vocab_size, trg_vocab_size):
        '''
        Transformer block.

        Arguments: 
            src_vocab_size (int): defined vocab size of src. 
            trg_vocab_size (int): defined vocab size of trg. 
        
        Inputs:
            src_input (tensor): [B, L]. 
            trg_input (tensor): [B, L]. 
            e_mask (tensor): [B, 1, L]. 
            d_mask (tensor): [B, L, L]. 

        Outputs:
            output (tensor): [B, L, trg_vocab_size]
        '''
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, d_model)
        self.positional_encoder = PositionalEncoder()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output_linear = nn.Linear(d_model, self.trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_input, trg_input, e_mask=None, d_mask=None):
        src_input = self.src_embedding(src_input) # (B, L) => (B, L, d_model)
        trg_input = self.trg_embedding(trg_input) # (B, L) => (B, L, d_model)
        src_input = self.positional_encoder(src_input) # (B, L, d_model) => (B, L, d_model)
        trg_input = self.positional_encoder(trg_input) # (B, L, d_model) => (B, L, d_model)

        e_output = self.encoder(src_input, e_mask) # (B, L, d_model)
        d_output = self.decoder(trg_input, e_output, e_mask, d_mask) # (B, L, d_model)

        output = self.softmax(self.output_linear(d_output)) # (B, L, d_model) => # (B, L, trg_vocab_size)

        return output


class Encoder(nn.Module):
    '''
    Encoder block that consist with number of num_layers EncoderLayer. 
    '''
    def __init__(self):
        '''
        Encoder block that consist with number of num_layers EncoderLayer. 

        Inputs:
            x (tensor): [B, L, d_model]. 
            e_mask (tensor): [B, 1, L]. 
        
        Outputs:
            x (tensor) [B, L, d_model]. 
        '''
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, e_mask):
        for i in range(num_layers):
            x = self.layers[i](x, e_mask)

        return self.layer_norm(x)


class Decoder(nn.Module):
    '''
    Decoder block that consist with number of num_layers DecoderLayer. 
    '''
    def __init__(self):
        '''
        Decoder block that consist with number of num_layers DecoderLayer. 

        Inputs:
            x (tensor): [B, L, d_model]. 
            e_output (tensor): [B, L, d_model]. 
            e_mask (tensor): [B, 1, L]. 
            d_mask (tensor): [B, L, L]. 
        
        Outputs:
            x (tensor) [B, L, d_model]. 
        '''
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        return self.layer_norm(x)   