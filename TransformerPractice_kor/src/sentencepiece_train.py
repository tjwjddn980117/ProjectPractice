from conf import *
from tqdm import tqdm

import os
import sentencepiece as spm

train_frac = 0.8

def train_sp(is_src=True):
    '''
    function for training the spm the sentences. 'this_model_prefix' will be the information about spm. 

    Inputs:
        is_src (bool): check the file with SRC/RAW. 
    
    '''
    template = "--input={} \
                --pad_id={} \
                --bos_id={} \
                --eos_id={} \
                --unk_id={} \
                --model_prefix={} \
                --vocab_size={} \
                --character_coverage={} \
                --model_type={}"

    if is_src:
        this_input_file = f"{DATA_DIR}/{SRC_RAW_DATA_NAME}"
        this_model_prefix = f"{SP_DIR}/{src_model_prefix}"
    else:
        this_input_file = f"{DATA_DIR}/{TRG_RAW_DATA_NAME}"
        this_model_prefix = f"{SP_DIR}/{trg_model_prefix}"

    config = template.format(this_input_file,
                            pad_id,
                            sos_id,
                            eos_id,
                            unk_id,
                            this_model_prefix,
                            sp_vocab_size,
                            character_coverage,
                            model_type)

    print(config)

    if not os.path.isdir(SP_DIR):
        os.mkdir(SP_DIR)

    print(spm)
    spm.SentencePieceTrainer.Train(config)
    
def split_data(raw_data_name, data_dir):
    '''
    split data with valid set and train set. 

    Inputs:
        raw_data_name (str): the name of raw_data file. ex) 'raw_data.src', 'raw_data.trg'
        data_dir (str): the name of data dir that save the split data of valid sets and train sets. 
    '''
    with open(f"{DATA_DIR}/{raw_data_name}") as f:
        lines = f.readlines()    
    
    print("Splitting data...")
    
    train_lines = lines[:int(train_frac * len(lines))]
    valid_lines = lines[int(train_frac * len(lines)):]
    
    if not os.path.isdir(f"{DATA_DIR}/{data_dir}"):
        os.mkdir(f"{DATA_DIR}/{data_dir}")
    
    with open(f"{DATA_DIR}/{data_dir}/{TRAIN_NAME}", 'w') as f:
        for line in tqdm(train_lines):
            f.write(line.strip() + '\n')
            
    with open(f"{DATA_DIR}/{data_dir}/{VALID_NAME}", 'w') as f:
        for line in tqdm(valid_lines):
            f.write(line.strip() + '\n')
            
    print(f"Train/Validation data saved in {DATA_DIR}/{data_dir}.")


if __name__=='__main__':
    train_sp(is_src=True)
    train_sp(is_src=False)
    split_data(SRC_RAW_DATA_NAME, SRC_DIR)
    split_data(TRG_RAW_DATA_NAME, TRG_DIR)