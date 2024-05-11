from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    '''
    Default Dataset for Competition for Image dataset. 
    '''
    def __init__(self, df, path_col,  mode='train'):
        '''
        Init for default Image Dataset. 

        Arguments:
            df (pd.csv): dataframe of csv. 
            path_col (str): the name of column about the information about data paths. 
            mode (str): the mode of dataset. 

        Inputs:
        
        '''
        self.df = df
        self.path_col = path_col
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.mode == 'train':
            row = self.df.iloc[idx]
            image = read_image(row[self.path_col])/256.
            label = row['class']
            data = {
                'image':image,
                'label':label
            }
            return data
        elif self.mode == 'val':
            row = self.df.iloc[idx]
            image = read_image(row[self.path_col])/256.
            label = row['class']
            data = {
                'image':image,
                'label':label
            }
            return data
        elif self.mode == 'inference':
            row = self.df.iloc[idx]
            image = read_image(row[self.path_col])/256.
            data = {
                'image':image,
            }
            return data

    def train_transform(self, image):
        pass