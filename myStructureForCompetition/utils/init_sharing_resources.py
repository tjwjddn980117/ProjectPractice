import pandas as pd
import os
import wandb
from sklearn.preprocessing import LabelEncoder
import joblib

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
    wandb.init(project="Example_Wandb", entity="CodingSlave")
    wandb_run_id = wandb.run.id

    my_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(my_dir, file_name)

    # 실행 ID를 파일에 저장
    with open("2024_bird_Wandb_id.txt", "w") as f:
        f.write(wandb_run_id)

def load_wandb_id():
    '''

    '''