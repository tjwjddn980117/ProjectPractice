from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from conf import *

import torch
import sentencepiece as spm
import numpy as np

src_sp = spm.SentencePieceProcessor()
trg_sp = spm.SentencePieceProcessor()
src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
trg_sp.Load(f"{SP_DIR}/{trg_model_prefix}.model")

def get_data_loader(file_name):
    '''
    get the data with 'file_name', then return with DataLoader. 
    data will open with /DATA_DIR/SRC_DIR/file_name and /DATA_DIR/TRG_DIR/file_name. 

    Inputs:
        file_name (str): the name of input file name. 
    
    Returns:
        dataloader (DataLoader): tuple(src_data, input_trg_data, output_trg_data). 
            src_data (tensor): [B, seq_len]. src_encoder + eos_id + (pad_id). 
            input_trg_data (tensor): [B, seq_len]. sos_id + trg_encoder + (pad_id).
            output_trg_data (tensor): [B, seq_len]. trg_encoder + sos_id + (pad_id). 

    '''
    print(f"Getting source/target {file_name}...")
    with open(f"{DATA_DIR}/{SRC_DIR}/{file_name}", 'r') as f:
        src_text_list = f.readlines()

    with open(f"{DATA_DIR}/{TRG_DIR}/{file_name}", 'r') as f:
        trg_text_list = f.readlines()

    print("Tokenizing & Padding src data...")
    src_list = process_src(src_text_list) # (sample_num, L)
    print(f"The shape of src data: {np.shape(src_list)}")

    print("Tokenizing & Padding trg data...")
    input_trg_list, output_trg_list = process_trg(trg_text_list) # (sample_num, L)
    print(f"The shape of input trg data: {np.shape(input_trg_list)}")
    print(f"The shape of output trg data: {np.shape(output_trg_list)}")

    dataset = CustomDataset(src_list, input_trg_list, output_trg_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def pad_or_truncate(tokenized_text):
    '''
    make the tokenized_text with defined 'seq_len' for same input/output size.

    Inputs:
        tokenized_text (list): [B, unknown lenght]. only tokenized sequence. 

    Returns:
        tokenized_text (list): [B, seq_len]. tokenized sequence with fixed length. 
    '''
    if len(tokenized_text) < seq_len:
        left = seq_len - len(tokenized_text)
        padding = [pad_id] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:seq_len]

    return tokenized_text

def process_src(text_list):
    '''
    processing the src. put the 'eos_id' into tokenized src sequence.

    Inputs:
        text_list (list): unprocessed sequence. 
    
    Returns:
        tokenized_list (list): sequence with tokenized + eos_id. 
    '''
    tokenized_list = []
    for text in tqdm(text_list):
        tokenized = src_sp.EncodeAsIds(text.strip())
        tokenized_list.append(pad_or_truncate(tokenized + [eos_id]))

    return tokenized_list

def process_trg(text_list):
    '''
    processing the trg. put the 'sos_id'/'eos_id' into tokenized trg sequence.

    Inputs:
        text_list (list): unprocessed sequence. 
    
    Returns:
        input_tokenized_list (list): sequence with sos_id + tokenized. 
        output_tokenized_list (list): sequence with tokenized + eos_id. 
    '''
    input_tokenized_list = []
    output_tokenized_list = []
    for text in tqdm(text_list):
        tokenized = trg_sp.EncodeAsIds(text.strip())
        trg_input = [sos_id] + tokenized
        trg_output = tokenized + [eos_id]
        input_tokenized_list.append(pad_or_truncate(trg_input))
        output_tokenized_list.append(pad_or_truncate(trg_output))

    return input_tokenized_list, output_tokenized_list

class CustomDataset(Dataset):
    '''
    CustomDataset for src_data / input_trg_data / output_trg_data. 
    '''
    def __init__(self, src_list, input_trg_list, output_trg_list):
        '''
        CustomDataset for src_data / input_trg_data / output_trg_data. 

        Outputs:
            src_data (tensor): [B, seq_len]. src_tokenized + eos_id. 
            input_trg_data (tensor): [B, seq_len]. sos_id + src_tokenized. 
            output_trg_data (tensor): [B, seq_len]. src_tokenized + eos_id. 
        '''
        super(CustomDataset).__init__()
        self.src_data = torch.LongTensor(src_list)
        self.input_trg_data = torch.LongTensor(input_trg_list)
        self.output_trg_data = torch.LongTensor(output_trg_list)

        assert np.shape(src_list) == np.shape(input_trg_list), "The shape of src_list and input_trg_list are different."
        assert np.shape(input_trg_list) == np.shape(output_trg_list), "The shape of input_trg_list and output_trg_list are different."

    def make_mask(self):
        '''
        making mask for input data. 

        Returns:
            e_mask (tensor[bool]): [B, 1, L]. make pad_id to FALSE and just unsqueeze the tensor.
            d_mask (tensor[bool]): [B, L, L]. make pad_id to FALSE and make each sequence a triangular matrix so that only the previous information is remembered for the input.
        '''
        e_mask = (self.src_data != pad_id).unsqueeze(1) # (num_samples, 1, L)
        d_mask = (self.input_trg_data != pad_id).unsqueeze(1) # (num_samples, 1, L)
        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool) # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask # (num_samples, L, L) padding false

        return e_mask, d_mask

    def __getitem__(self, idx):
        return self.src_data[idx], self.input_trg_data[idx], self.output_trg_data[idx]

    def __len__(self):
        return np.shape(self.src_data)[0]