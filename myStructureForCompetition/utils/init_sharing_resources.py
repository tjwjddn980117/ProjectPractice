import pandas as pd
import os
import wandb
from sklearn.preprocessing import LabelEncoder
import joblib

from .config import CFG

def init_label_encoder(csv_file_path, file_name):
    '''
    this function works for init the label with encoding space. 
    it load the file and save with 'file_name'+.pkl

    you can use this function like "init_label_encoder('../data/train.csv', '2024_bird')",
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

    if os.path.isfile(save_path+'.pkl'):
        print(f"Already we have the file about {file_name}")
        return

    csv_df = pd.read_csv(csv_path)
    le = LabelEncoder()
    csv_df['class'] = le.fit_transform(csv_df['label'])

    # LabelEncoder 저장
    joblib.dump(le, save_path+'.pkl')

def load_label_encoder(file_name):
    '''
    this function works for loading encoder of label.

    Inputs:
        file_name (str): the name of saving file. "load_label_encoder('2024_bird')"
    
    Returns:
        le (LabelEncoder): the encoded label. 
    '''
    my_dir = os.path.dirname(os.path.abspath(__file__))
    load_path = os.path.join(my_dir, file_name)
    load_path = load_path + '.pkl'

    if not os.path.isfile(load_path):
        print(f"There's no file name of {load_path}")
        return 
    le = joblib.load(load_path)

    return le

def init_wandb(file_name):
    '''
    this function works for init the wandb. 

    Inputs:
        file_name (str): the name of saving file. "init_wandb('bird_project_wandb')"

    Returns:
        _ 
    '''

    my_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(my_dir, file_name)

    save_path = save_path + '.txt'

    if os.path.isfile(save_path):
        print(f"Already we have the file about {file_name}")
        return
    
    # WandB 실행 시작
    wandb.init(project=CFG.WANDB_ID_NAME)
    wandb_run_id = wandb.run.id
    
    # 실행 ID를 파일에 저장
    with open(save_path, "w") as f:
        f.write(wandb_run_id)

def load_wandb_id(file_name):
    '''
    this function works for load the wandb.

    Inputs:
        file_name (str): the name of saving file. "load_wandb_id('bird_project_wandb')"
    
    Returns:
        _ 
    '''

    # check the wandb.run. 
    if wandb.run is not None:
        print("WandB is already initialized.")
        return
    
    # 실행 ID 불러오기
    my_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(my_dir, file_name)
    save_path = save_path + '.txt'

    run_id_path = os.path.join(os.path.dirname(__file__), save_path)

    if not os.path.isfile(run_id_path):
        print(f"There's no file name of {run_id_path}")
        return

    with open(run_id_path, "r") as f:
        wandb_run_id = f.read().strip()

    # WandB 실행에 연결
    wandb.init(project=CFG.WANDB_ID_NAME, id=wandb_run_id, resume="allow")

# init_wandb('test_wandb')
# load_wandb_id('test_wandb')

