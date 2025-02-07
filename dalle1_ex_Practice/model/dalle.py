import torch
import torch.nn as nn
from model.mingpt import GPT, DallEGPTConfig

class DallE(nn.Module):
    r"""
    Class handling the logic for DallE
    Calls the vae and passes the text and image tokens
    together with target to gpt
    """
    def __init__(self, vae, num_words, image_size, max_text_len, image_vocab_size, gpt_config):
        '''
        Class handling the logic for DallE
        Calls the vae and passes the text and image tokens
        together with target to gpt. 
        target은 self.training의 bool 차이를 두고 확인함. target은 input 되는 image를 그대로 씀. (비지도학습). 

        Arguments: 
            vae (nn.module): the model of vae. 
            num_words (int): the number of words. 
            image_size (int): the size of image. 
            max_text_len (int): the max lenght of text. 
            gpt_config (dict): the config of gpt. 
        
        Inputs:
            im (tensor): [B, C, H, W]. the tensor of image. 
            text (tensor): [B, text_t]. the tensor of text. 
        
        Outputs:
            logits (tensor): [B, T, config.text_vocab_size + config.image_vocab_size].
            loss_text (float): The loss between prediction and text. 
            loss_image (float): THe loss between prediction and image. 
        '''
        super(DallE, self).__init__()
        self.vae = vae
        
        # Text Vocab size
        self.num_words = num_words
        # Number of Image tokens
        self.image_size = image_size
        # Maximum Text Sequence Length
        self.max_text_len = max_text_len
        
        # Image tokens vocabulary size (num_of_embeddings)
        image_vocab_size = image_vocab_size

        # Length of largest sequence so that we tell gpt
        # to have that as the context size
        max_sequence_len = max_text_len + image_size*image_size
        config = DallEGPTConfig(text_vocab_size=num_words, 
                           image_vocab_size=image_vocab_size, 
                           max_sequence_len=max_sequence_len, 
                           im_size=image_size, 
                           **gpt_config)
        self.gpt = GPT(config)

    def forward(self, im, text):
        # Call Discrete vae
        # image_tokens.shape = [B, C * (H/8) * (W/8)]. 
        # flatten each image. 
        image_tokens = self.vae.get_codebook_indices(im).reshape(im.size(0), -1)

        # Shift the target image tokens as image tokens + text vocab size
        # ex) text vocab size가 1000이라면 기존에 갖고있는 image token에 text token을 더해서 image tokens를 미리 준비해놓는다. 
        # Last fc layer will predict 0 to (num_words + num_embeddings) output probabilities
        # We will formulate the target such first num_words-1 are text token probabilities
        # and num_words to num_words+num_embeddings are image token probabilities
        target_image_tokens = image_tokens + self.num_words
        labels = None
        
        if self.training:
            # Pass one position shifted tokens as targets only in training
            labels = torch.cat((text[:, 1:], target_image_tokens), dim=1)
        # Loss of text and Loss image separately so that we can get better images. 
        logits, loss_text, loss_image = self.gpt(image_tokens, text, targets=labels)
        return logits, loss_text, loss_image