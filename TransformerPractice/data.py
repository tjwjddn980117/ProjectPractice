from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

tokenizer = Tokenizer()
loader = DataLoader(ext=('.en', '.de'), 
                    tokenize_en=tokenizer.tokenize_en, 
                    tokenize_de=tokenizer.tokenize_de, 
                    init_token='<sos>',
                    eos_token='<eos>')

train, valid, test = loader.make_dataset()
loader.build_vocab(train_data=train,min_freq=2)
train_iter, valid_iter, test_iter = loader.make_iter(train=train, 
                                                     validate=valid, 
                                                     test=test,
                                                     batch_size=batch_size,
                                                     device=device)
    
# The source or target has the original text data. 
# And the vocab property of the source represents the vocabulary of that text data. 
# This vocabulary book has a dictionary form that maps each word to its own integer index.
src_pad_idx = loader.source.vocab.stoi('<pad>') # commonly 1
tar_pad_idx = loader.target.vocab.stoi('<pad>') # commonly 1
tar_sos_idx = loader.target.vocab.stoi('<sos>') # commonly 2

enc_voc_size = len(loader.source.vocab) # number of unique words in the source
dec_voc_size = len(loader.target.vocab) # number of unique words in the target