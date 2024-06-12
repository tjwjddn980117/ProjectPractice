import torch
from torchvision.transforms import v2 as  transforms
from torchvision.transforms import RandomAffine, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter

class CustomCollateFn:
    '''
    The calss for Collate Function. 

    Arguments:
        transform (transforms.Compose): Transform functions.
        mode (str): the mode of datasets. 
    '''
    def __init__(self, transform, mode):
        self.mode = mode
        self.transform = transform

    def __call__(self, batch):
        if self.mode=='train':
            pixel_values = torch.stack([self.transform(data['image']) for data in batch])
            label = torch.LongTensor([data['label'] for data in batch])
            return {
                'pixel_values':pixel_values,
                'label':label,
            }
        elif self.mode=='val':
            pixel_values = torch.stack([self.transform(data['image']) for data in batch])
            label = torch.LongTensor([data['label'] for data in batch])
            return {
                'pixel_values':pixel_values,
                'label':label,
            }
        elif self.mode=='inference':
            pixel_values = torch.stack([self.transform(data['image']) for data in batch])
            return {
                'pixel_values':pixel_values,
            }
        
train_transform = transforms.Compose([
    transforms.Resize(size=(196,196), interpolation=transforms.InterpolationMode.BICUBIC),
    RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

val_transform = transforms.Compose([
    transforms.Resize(size=(196,196), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

test_transform = transforms.Compose([
    transforms.Resize(size=(256,256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

train_collate_fn = CustomCollateFn(train_transform, 'train')
val_collate_fn = CustomCollateFn(val_transform, 'val')
test_collate_fn = CustomCollateFn(test_transform, 'inference')

def get_collate_fn():
    '''
    The function to get collate function for training and validation. 

    Returns:
        train_collate_fn (CollateFn): CollateFn for training set.
        val_collate_fn (CollateFn): CollateFn for validation set. 
    '''
    return train_collate_fn, val_collate_fn