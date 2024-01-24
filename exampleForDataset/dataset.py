import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from conf import *

class JigsawDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        csv_file (string): csv file with all the images and labels.
        transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_frame = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(DATA_PATH, self.data_frame.iloc[idx, 1])
        image = Image.open(img_name)
        labels = self.data_frame.iloc[idx, 2:].astype(int).tolist()

        if self.transform:
            image = self.transform(image)
        
        return (image, labels)

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # resize images to 512 x 512
    transforms.ToTensor(),  # convert PIL image to tensor
])

# Initialize dataset
TRAIN_DATASET = JigsawDataset(csv_file=pd.read_csv(TRAIN_CSV), transform=transform)
TEST_DATASET = JigsawDataset(csv_file=pd.read_csv(TEST_CSV), transform=transform)

# Initialize data loader
TRAIN_DATA_LOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True)
TEST_DATA_LOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=True)