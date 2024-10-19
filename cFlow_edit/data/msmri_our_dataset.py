import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torch as th
from skimage import io, transform
import random
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset

class MS_MRI_Dataloader():
    def __init__(self, exp_config):

        # self.train_ds = MS_MRI(os.path.join(exp_config.data_folder, 'train')
        #                                    , exp_config.transform['train'], test_flag=False)
        # self.val_ds = MS_MRI(os.path.join(exp_config.data_folder, 'val'),
        #                                  exp_config.transform['val'], test_flag=False)

        # Load the entire 'training' dataset
        full_train_ds = MS_MRI(os.path.join(exp_config.data_folder, 'training'),
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

        self.test_ds = MS_MRI(os.path.join(exp_config.data_folder, 'testing'),
                                          exp_config.transform['test'], test_flag=True)

        self.train = DataLoader(self.train_ds, shuffle=True, batch_size=exp_config.train_batch_size,
                                drop_last=True, pin_memory=True, num_workers=exp_config.num_w)

        self.validation = DataLoader(self.val_ds, shuffle=False, batch_size=exp_config.val_batch_size,
                                drop_last=True, pin_memory=True, num_workers=exp_config.num_w)

        self.test = DataLoader(self.test_ds, shuffle=False, batch_size=exp_config.test_batch_size,
                                drop_last=False, pin_memory=True, num_workers=exp_config.num_w)

class MS_MRI(torch.utils.data.Dataset):
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
        
        self.seqtypes = ['image-flair', 'image-mprage', 'image-pd', 'image-t2', 'label1', 'label2']

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
                    datapoint[seqtype] = os.path.join(root, f)
                    #print(datapoint)
                # print("Datapoint keys:", datapoint.keys())
                # print("Expected keys:", self.seqtypes_set)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)
                

    def __getitem__(self, x):
        filedict = self.database[x]
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            img = io.imread(filedict[seqtype])
            img = transform.resize(img, (128, 128))
            if not seqtype == 'label1' and not seqtype == 'label2':
                img = img / 255
            path=filedict[seqtype] # slice_ID = path[0].split("/", -1)[2]
            out.append(torch.tensor(img))
        out = torch.stack(out)
        
        flair = torch.unsqueeze(out[0], 0)
        mprage = torch.unsqueeze(out[1], 0)
        pd = torch.unsqueeze(out[2], 0)
        t2 = torch.unsqueeze(out[3], 0)
    
        image = torch.cat((flair, mprage, pd, t2), 0)
        image = image.type(torch.FloatTensor)
        
        # out에서 랜덤하게 label을 4번 뽑아서 [4, 128, 128]로 만들기
        labels = []
        for _ in range(4):
            # 4 또는 5 중에서 랜덤하게 선택
            label = out[random.randint(4, 5)]
            # 차원 추가: [128, 128] -> [1, 128, 128]
            label = torch.unsqueeze(label, 0)   
            labels.append(label)

        # labels 리스트를 [4, 128, 128]로 합치기
        labels = torch.cat(labels, dim=0)
        labels = labels.type(torch.FloatTensor)
        
        path = path.replace('\\','/')

        labels_prob = np.ones(labels.shape[0], dtype = np.float32) / labels.shape[0]
        
        return image, labels, labels_prob, path

    def __len__(self):
        return len(self.database)