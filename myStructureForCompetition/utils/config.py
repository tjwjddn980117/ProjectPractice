import os
import torch

class CFG:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_DEVICES = torch.cuda.device_count()
    NUM_WORKERS = os.cpu_count()
    NUM_CLASSES = 25
    N_SPLIT = 5
    EPOCHS = 16
    BATCH_SIZE = (
        32 if torch.cuda.device_count() < 2 
        else (32 * torch.cuda.device_count())
    )
    LR = 0.001
    APPLY_SHUFFLE = True
    SEED = 768
    HEIGHT = 224
    WIDTH = 224
    CHANNELS = 3
    IMAGE_SIZE = (224, 224, 3)
    
    # Define paths
    #DATASET_PATH = "/content/drive/MyDrive/Colab Notebooks/dataset"
    #TRAIN_PATH = '/content/drive/MyDrive/Colab Notebooks/dataset/train/'
    #TEST_PATH = '/content/drive/MyDrive/Colab Notebooks/dataset/test'