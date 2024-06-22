import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import CFG
from utils.init_sharing_resources import init_label_encoder, init_wandb
from trainers.pytorch_trainer import trainer
import torch

import warnings

warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")

cvs_file_path = CFG.PROJECT_PATH + 'data\\train.csv'
init_label_encoder(cvs_file_path, CFG.LABEL_ENCODER_NAME)   
init_wandb(CFG.WANDB_ID_NAME)

torch.autograd.set_detect_anomaly(True)

trainer(load_file='C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\myStructureForCompetition\\checkpoints\\fold0_epoch19.pt')
#trainer()