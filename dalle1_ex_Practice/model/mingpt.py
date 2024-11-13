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