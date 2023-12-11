import torch
from functools import partial
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2, WikiText103

from constants import *

# the function just define 'get_english_tokenizer' for 'get_tokenizer'
def get_english_tokenizer():
    tokenizer = get_tokenizer("basic_english", language="en")
    return tokenizer

# the function get data with style WikiText2/WikiText103
def get_data_iterator(ds_name, ds_type, data_dir):
    if ds_name == "WikiText2":
        data_iter = WikiText2(root=data_dir, split=(ds_type))
    elif ds_name == "WikiText103":
        data_iter = WikiText103(rood=data_dir, split=(ds_type))
    else:
        raise ValueError("Choose dataset from: WikiText2, WikiText103")
    data_iter = to_map_style_dataset(data_iter)
    return data_iter

# this function get vacab from data_iter
def build_vocab(data_iter, tokenizer: function):
    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter), # mapping 'data_iter' to 'tokenizer' function.
        specials=["<unk>"], # just put <unk> in the 'vocab'
        min_freq=MIN_WORD_FREQUENCY, # world should more than this frequency
    )
    vocab.set_default_index(vocab["<unk>"])
    # list of kind of vocab
    return vocab # save with index. ( List[int] )

# if CBoW, this function will used
# CBoW is the function that predice Center word from Context word
def collate_cbow(batch, text_pipeline: function):
    """
    Collate_fn for CBOW model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=CBOW_N_WORDS past words 
    and N=CBOW_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
    Each element in `batch_output` is a middle word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        # this 'text_pipeline' is lambda x: vocab(tokenizer(x))
        # the function work that 'text to index'
        # this can possible because vocab is type of dictionary. (let's check it. that's not a function)
        text_tokens_ids = text_pipeline(text)

        # text_token_index should inner in whole inedex
        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        # resize text_tokens_index
        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]
        
        # Let's say text_tokens_ids is set to '1234567' and CBOW_N_WORDS is set to 2
        # for each iter, token_id_sequence will '12345', '23456', 34567'
        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)
        
        # batch_input will [[1,2,4,5], [2,3,5,6], [3,4,6,7]]
        # batch_output will [3, 4, 5]
    
    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output

def collate_skipgram(batch, text_pipeline):
    """
    Collate_fn for Skip-Gram model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=SKIPGRAM_N_WORDS past words 
    and N=SKIPGRAM_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is a middle word.
    Each element in `batch_output` is a context word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        # this 'text_pipeline' is lambda x: vocab(tokenizer(x))
        # the function work that 'text to index'
        # this can possible because vocab is type of dictionary. (let's check it. that's not a function)
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
            # input wil int
            # outputs will [1,2,4,5]
            input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
            outputs = token_id_sequence

            for output in outputs:
                # batch_input will [3,3,3,3,4,4,4,4,5,5,5,5]
                # batch_output will [1,2,4,5,2,3,5,6,3,4,6,7]
                batch_input.append(input_)
                batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output

def get_dataloader_and_vocab(
    model_name, ds_name, ds_type, data_dir, batch_size, shuffle, vocab=None
):

    data_iter = get_data_iterator(ds_name, ds_type, data_dir)
    tokenizer = get_english_tokenizer()

    if not vocab:
        vocab = build_vocab(data_iter, tokenizer)
    
    # it change tokenizer to index
    text_pipeline = lambda x: vocab(tokenizer(x))

    if model_name == "cbow":
        collate_fn = collate_cbow
    elif model_name == "skipgram":
        collate_fn = collate_skipgram
    else:
        raise ValueError("Choose model from: cbow, skipgram")

    # DataLoader class is basicaly type of iter class
    # data_iter is full dataseet
    # class split the data_iter with batch_size
    # and collate_fn can resize batched data.
    # it could run with just one parameter(collate_fn) (basicaly, collate_fn need two parameter)
    #   because DataLoader can automaticaly give partial an iter_data (batched data)
    dataloader = DataLoader(
        data_iter,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, text_pipeline=text_pipeline),
    )
    return dataloader, vocab