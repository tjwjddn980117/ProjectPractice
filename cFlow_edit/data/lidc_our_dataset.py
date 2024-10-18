import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torch as th
from skimage import io
import random
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset

class lidc_Dataloader():
    def __init__(self, exp_config):

        # self.train_ds = LIDC_IDRI(os.path.join(exp_config.data_folder, 'train')
        #                                    , exp_config.transform['train'], test_flag=False)
        # self.val_ds = LIDC_IDRI(os.path.join(exp_config.data_folder, 'val'),
        #                                  exp_config.transform['val'], test_flag=False)

        # Load the entire 'training' dataset
        full_train_ds = LIDC_IDRI(os.path.join(exp_config.data_folder, 'training'),
                                  exp_config.transform['train'], test_flag=False)
        
        # Split 'training' data into training and validation sets
        total_len = len(full_train_ds)
        train_len = int(0.8 * total_len)  # 80% for training
        val_len = total_len - train_len  # 20% for validation

        # Generate random indices for splitting
        indices = list(range(total_len))
        random.shuffle(indices)

        train_indices = indices[:train_len]
        val_indices = indices[train_len:]

        # Create subsets for training and validation
        self.train_ds = Subset(full_train_ds, train_indices)
        self.val_ds = Subset(full_train_ds, val_indices)

        self.test_ds = LIDC_IDRI(os.path.join(exp_config.data_folder, 'testing'),
                                          exp_config.transform['test'], test_flag=True)

        self.train = DataLoader(self.train_ds, shuffle=True, batch_size=exp_config.train_batch_size,
                                drop_last=True, pin_memory=True, num_workers=exp_config.num_w)

        self.validation = DataLoader(self.val_ds, shuffle=False, batch_size=exp_config.val_batch_size,
                                drop_last=True, pin_memory=True, num_workers=exp_config.num_w)

        self.test = DataLoader(self.test_ds, shuffle=False, batch_size=exp_config.test_batch_size,
                                drop_last=False, pin_memory=True, num_workers=exp_config.num_w)

class LIDC_IDRI(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['image', 'label0', 'label1', 'label2', 'label3']
        else:
            self.seqtypes = ['image', 'label0', 'label1', 'label2', 'label3']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[0]
                    #print(seqtype)
                    datapoint[seqtype] = os.path.join(root, f)
                    #print(datapoint)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)
                

    def __getitem__(self, index):
        filedict = self.database[index]
        out = []

        # Load image and all labels
        image = io.imread(filedict['image']) / 255
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Shape: [1, H, W]
        
        # Load all labels (label0 to label3)
        labels = []
        for i in range(4):
            label = io.imread(filedict[f'label{i}']) / 255
            label = torch.tensor(label, dtype=torch.float32)
            labels.append(label)
            
        
        # Stack labels into a single tensor
        labels = torch.stack(labels, dim=0)  # Shape: [4, H, W]

        # Create a sample dictionary
        # sample = {
        #     'image': image,
        #     'label': labels
        # }
    # 
        # # Apply transform if available
        # if self.transform is not None:
        #     sample = self.transform(sample)
        #     image = sample['image']
        #     labels = sample['label']

        labels_prob = np.ones(labels.shape[0], dtype = np.float32) / labels.shape[0]

        # Path for reference or identification
        path = filedict['image'].replace('\\', '/')
        
        return image, labels, labels_prob, path

    def __len__(self):
        return len(self.database)