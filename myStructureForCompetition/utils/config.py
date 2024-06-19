import os
import torch

class CFG:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_DEVICES = torch.cuda.device_count()
    NUM_WORKERS = os.cpu_count()
    NUM_CLASSES = 25
    N_SPLIT = 5
    EPOCHS = 20
    BATCH_SIZE = (
        18 if torch.cuda.device_count() < 2 
        else (18 * torch.cuda.device_count())
    )
    LR = 0.001
    APPLY_SHUFFLE = True
    SEED = 768
    HEIGHT = 196
    WIDTH = 196
    CHANNELS = 3
    IMAGE_SIZE = (196, 196, 3)
    
    LABEL_ENCODER_NAME = 'test_encoder'
    WANDB_ID_NAME = 'test_wandb'
    
    PROJECT_PATH = 'C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\myStructureForCompetition\\'
    CHECKPOINT_PATH = ''
    
    # Define paths
    #DATASET_PATH = "/content/drive/MyDrive/Colab Notebooks/dataset"
    #TRAIN_PATH = '/content/drive/MyDrive/Colab Notebooks/dataset/train/'
    #TEST_PATH = '/content/drive/MyDrive/Colab Notebooks/dataset/test'
