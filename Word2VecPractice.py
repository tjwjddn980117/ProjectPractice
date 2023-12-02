# reference https://yonghee.io/word-embedding-pytorchword2vec/
from typing import *
from itertools import chain

import torch
import torch.nn.functional as F
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from torch.autograd import Variable

class BagOfWord:
    def __init__(self):
        lemm = WordNetLemmatizer()
        # it could find the origin of the word
        self.lemmatize = lemm.lemmatize
        return
    


    # 1. this function is for tokenize
    def make_tokenized_matrix_english(self, texts: List[str], lemmatize=True) -> List[List]:
        if lemmatize:
            self.tokenized_matrix = [[self.lemmatizer(word) for word in word_tokenize(text)] for text in texts]
        else:
            self.tokenized_matrix = [word_tokenize(text) for text in texts]
        return self.tokenized_matrix
    
    # 2. this function is for tokenize to index
    def get_uniques_from_nested_lists(nested_lists: List[List]) -> List:
        uniques = {}

    def make_token_to_idx(self):
        return
    
    # 3. this function is for get windows pair
    def get_window_pair(self, tokens: List[str], win_size=4, as_index=True) -> List[Tuple]:
        return
    
    # 4. this function is for making pair matrix
    def make_pair_matrix(self, win_size, as_index=True):
        return
