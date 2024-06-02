import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import CFG
from utils.init_sharing_resources import init_label_encoder, init_wandb
from trainers.pytorch_trainer import trainer

cvs_file_path = CFG.PROJECT_PATH + 'data\\train.csv'
init_label_encoder(cvs_file_path, CFG.LABEL_ENCODER_NAME)
init_wandb(CFG.WANDB_ID_NAME)

trainer()