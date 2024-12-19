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
        layer_past (bool): not used. I guess that it might prevent pre-trained model. 
    
    Outputs:
        y (tensor): [B, T, C]. B is batch size, T is sequence lenght, C is embedding dimensionality (n_embd). 
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
    
class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        '''
        The class of Block. 

        Arguments:
            config (dict): the dictionary of configs. 
        
        Inputs:
            x (tensor): [B, T, C]. 
        
        Outputs:
            x (tensor): [B, T, C]. 
        '''
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        # x = [B, T, C]. 
        y = self.attn(self.ln1(x))
        # y = [B, T, C]. 
        x = x + y
        # x = [B, T, C]. 
        x = x + self.mlp(self.ln2(x))
        # x = [B, T, c]. 
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        '''
        the full GPT language model, with a context size of block_size 

        Arguments: 
            image_tokens (tensor): [b, im_t]. the token of image. 
            text_tokens (tensor): [b, text_t]. the token of text. 
            targets (tensor): [b, text_t]. the tensor of targets. if you validate the target, the targets should be None. 

        Inputs:
            image_tokens (tensor): [B, im_t]. 
            tensor_tokens (tensor): [B, text_t]. 
            targets (tensor): [B, T]. 
        
        Outputs:
            logits (tensor):
            loss_text (float): 
            loss_image (float): 
        '''
        super().__init__()

        # input embedding stem
        self.text_tok_emb = nn.Embedding(config.text_vocab_size, config.n_embd)
        self.image_tok_emb = nn.Embedding(config.image_vocab_size, config.n_embd)
        
        self.text_pos_emb = nn.Parameter(torch.zeros(1, config.num_text_tokens, config.n_embd))
        self.image_pos_emb = nn.Parameter(torch.zeros(1, config.im_size ** 2, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.text_vocab_size + config.image_vocab_size, bias=False)
        self.config = config
        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        '''
        the function get config block size. 

        Inputs:
            _ (): _
        
        Outputs:
            self.block_size (int): the size of config block size. 
        '''
        return self.block_size

    def _init_weights(self, module):
        '''
        the function for init the weight. 
        
        Inputs:
            module (nn.model): the model for init weights. 
        '''
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, GPT):
            torch.nn.init.normal_(module.text_pos_emb, mean=0.0, std=0.02)
            # Simply do the same thing for image as for text
            torch.nn.init.normal_(module.image_pos_emb, mean=0.0, std=0.02)
  

    def forward(self, image_tokens, text_tokens, targets=None,):
        b, im_t = image_tokens.size()
        b, text_t = text_tokens.size()
        assert im_t + text_t <= self.block_size, "Cannot forward, model block size is exhausted."

        # self.text_tok_emb = nn.Embedding(config.text_vocab_size, config.n_embd)
        # text_tok_emb =  (text_vocab_size, n_embd) of table size. 
        # text_emb.shape = [b, text_t, n_embd].     
        text_emb = self.text_tok_emb(text_tokens)
        # self.text_pos_emb = nn.Parameter(torch.zeros(1, config.num_text_tokens, config.n_embd))
        # text_pos_emb.shape = [1, num_text_tokens, n_embd]. 
        # text_pos.shape = [1, text_t, n_embd]. 
        text_pos = self.text_pos_emb[:, :text_t, :]
        # text_token_embeddings.shape = [b, text_t, n_embd]. 
        text_token_embeddings = self.drop(text_emb + text_pos)
        x = text_token_embeddings
        
        # Add image tokens for input sequence if needed.
        # Won't be needed for first pixel generation
        if im_t > 0:
            # self.image_tok_emb = nn.Embedding(config.image_vocab_size, config.n_embd)
            # image_tok_emb = (image_vocab_size, n_embd) of table size. 
            # image_emb = [b, im_t, n_embc]. 
            image_emb = self.image_tok_emb(image_tokens)
            # self.image_pos_emb = nn.Parameter(torch.zeros(1, config.im_size**2, config.n_embd))
            # image_pos_emb.shape = [1, im_size**2, n_embd]. 
            # image_pos.shape = [1, im_t, n_embd]. 
            image_pos = self.image_pos_emb[:, :im_t, :]
            # image_token_embeddings.shape = [b, im_t, n_embd]. 
            image_token_embeddings = self.drop(image_emb + image_pos)
            # x = [b, text_t+im_t, n_embd]. 
            x = torch.cat([x, image_token_embeddings], dim=1)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        # if we are given some desired targets also calculate the loss
        # Separate text and image loss
        loss_text = None
        loss_image = None
        if targets is not None:
            # logits.shape = [b, (seq_lenght - 1), (text_vocab_size + image_vocab_size)]. 
            logits = logits[:, :-1, :]
            
            # Separate text and image token loss computation
            # text_t 까지만 체크를 해야함. 
            # text_logits.shape = [b, text_vocab_size + image_vocab_size, text_t - 1]. 
            # image_logits.shape = [b, text_vocab_size + image_vocab_size, seq_length - text_t].    
            text_logits = logits[:, :text_t - 1, :].permute((0, 2, 1))
            image_logits = logits[:, text_t - 1:, :].permute((0, 2, 1))
            
            # For now just mask logits of image tokens for text targets
            # And mask out text tokens logits for iamge targets
            # Dont want gpt to gain points by simply decreasing scores for indexes of the other type
            # And anyway at inference you would always sample image token when generating image
            text_logits[:, self.config.text_vocab_size:, :] = -torch.finfo(logits.dtype).max
            image_logits[:, :self.config.text_vocab_size, :] = -torch.finfo(logits.dtype).max
            loss_text = F.cross_entropy(text_logits, targets[:, :text_t-1])
            loss_image = F.cross_entropy(image_logits, targets[:, text_t-1:])
        return logits, loss_text, loss_image