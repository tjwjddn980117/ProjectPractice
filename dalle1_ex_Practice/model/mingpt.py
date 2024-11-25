"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class DallEGPTConfig:
    r"""
    Minimal DallE config holding all fields requored for
    training gpt for both text and image tokens. 
    """
    def __init__(self, text_vocab_size,
                 image_vocab_size,
                 max_sequence_len, im_size, **kwargs):
        '''
        Minimal DallE config holding all fields requored for training GPT for both text and image tokens. 

        Arguments:
            text_vocab_size (int): the lenght of text description. 
            image_vocab_size (int): the lenght of image description. 
            max_sequence_len (int): the block size. 
            im_size (int): the size of image. 
            **kwargs (additional parameters): additional parameters. 
        
        HowToUse:
            config = DallEGPTConfig(
            text_vocab_size=1000,
            image_vocab_size=500,
            max_sequence_len=128,
            im_size=14,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            n_layer=4,
            n_head=8,
            n_embd=512
            )

            config = DallEGPTConfig(
            text_vocab_size=1000,
            image_vocab_size=500,
            max_sequence_len=128,
            im_size=14,
            **config.model_params
            )
        '''
        self.text_vocab_size = text_vocab_size
        self.image_vocab_size = image_vocab_size
        # Fixing block size to maximum sequence length we have seen
        self.block_size = max_sequence_len
        self.im_size = im_size
        self.num_text_tokens = max_sequence_len - im_size*im_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.

    Arguments:
        config (dict): the dictionary of configs. 

    Inputs:
        x (tensor): [B, T, C]. B is batch size, T is sequence lenght, C is embedding dimensionality (n_embd). 
        layer_past (bool): 
    
    Outputs:

    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
    
    def forward(self, x, layer_past=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y