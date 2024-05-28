import pandas as pd
import os
import wandb
from sklearn.preprocessing import LabelEncoder
import joblib

from config import WANDB

def init_label_encoder(csv_file_path, file_name):
    '''
    this function works for init the label with encoding space. 
    it load the file and save with 'file_name'+.pkl

    you can use this function like init_label_encoder('../data/train.csv', '2024_bird'),
    then the return will be '2024_bird.pkl'

    Inputs:
        csv_file_path (str): the path of csv file. 
        file_name (str): the name of saving file. 
    
    Returns:
        _
    '''
    my_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(my_dir, csv_file_path)
    save_path = os.path.join(my_dir, file_name)

    csv_df = pd.read_csv(csv_path)
    le = LabelEncoder()
    csv_df['class'] = le.fit_transform(csv_df['label'])

    # LabelEncoder 저장
    joblib.dump(le, save_path+'.pkl')

def load_label_encoder(file_name):
    '''
    this function works for loading encoder of label.

    Inputs:
        file_name (str): the name of saving file. ('2024_bird.pkl')
    
    Outputs:
        le (LabelEncoder): the encoded label. 
    '''
    my_dir = os.path.dirname(os.path.abspath(__file__))
    load_path = os.path.join(my_dir, file_name)
    le = joblib.load(load_path)

    return le

def init_wandb(file_name):
    '''
    this function works for init the wandb. 

    '''
    # WandB 실행 시작
    wandb.init(project=WANDB.PROJECT_NAME)
    wandb_run_id = wandb.run.id

    my_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(my_dir, file_name)

    save_path = save_path + '.txt'
    # 실행 ID를 파일에 저장
    with open(save_path, "w") as f:
        f.write(wandb_run_id)

def load_wandb_id(file_name):
    '''
    this function works for load the wandb.

    '''
    # 실행 ID 불러오기
    my_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(my_dir, file_name)
    save_path = save_path + '.txt'

    run_id_path = os.path.join(os.path.dirname(__file__), save_path)
    with open(run_id_path, "r") as f:
        wandb_run_id = f.read().strip()

    print(wandb_run_id)
    # WandB 실행에 연결
    wandb.init(project=WANDB.PROJECT_NAME, id=wandb_run_id, resume="allow")

init_wandb('test_wandb')
load_wandb_id('test_wandb')