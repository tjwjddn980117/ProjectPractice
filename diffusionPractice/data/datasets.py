from torch import nn
from pathlib import Path
from functools import partial
from torchvision import transforms as T

from PIL import Image
from torch.utils.data import Dataset

from ..utils.functions import convert_image_to_fn, exists

# dataset classes
class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        '''
        the class for Datasets.

        Arguments:
            folder (str): the path about datasets. 
            image_size (int): the size of image. 
            exts (list([str])): supported type of image. 
            augment_horizontal_flip (bool): choose the horizontal flip of image. 
            convert_image_to (function): the function for convert image type. 

        Inputs:
            index (int): the index of image. 
        
        Outputs:
            img (Image): a single image with tranform. 
        '''
        super(Dataset).__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
