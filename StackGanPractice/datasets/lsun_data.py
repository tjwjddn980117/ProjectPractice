from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.utils.data as data
import torchvision.transforms as transforms
import os
import os.path
import pickle
from ..miscc.config import cfg
from utils import get_imgs

import torch.utils.data as data

import six

######################
###   LSUN Data    ###
######################
class LSUNClass(data.Dataset):
    def __init__(self, db_path, base_size=64,
                 transform=None, target_transform=None):
        '''
        class about LSUNC data (Large-scale Scene Understanding data).
    
        Attributes:
            db_path (str): it is the path of database.
            env ( ): it is the environment o lmdb.
            lenght (int): it is the number of all instance items.
            keys ( ): load LSUNC keys.
    
            transform ( ): information to define of transform.
            target_transform ( ): information to define of target_transform.
            norm (transforms): transforms for narmalize.
    
            imsize (list): list of image size [64, 128, 256, ...].
        '''
        import lmdb
        # Storing LMDB data path
        self.db_path = db_path
        # Store the LMDB environment, 
        #  which is used to access the database.
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        # Stores the number of items stored in the database.
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
            print('length: ', self.length) # number of items.

        # approach with cache file
        cache_file = db_path + '/cache'
        # Store the number of keys stored in the database.
        # If cache_file is exist, Load the cache
        #  but it wasn't, store cache_file
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
            print('Load:', cache_file, 'keys: ', len(self.keys))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))
            
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.transform = transform
        self.target_transform = target_transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def __getitem__(self, index):
        '''
        call the data.

        Arguments:
            index (int): index of image. we get images from imgbuf.
            
        Returns:
            imgs_list (list[Image]): list of resized images data type is (Image).
        '''
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        imgs = get_imgs(buf, self.imsize,
                        transform=self.transform,
                        normalize=self.norm)
        return imgs
    
    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'