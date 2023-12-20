from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import os.path
import pickle
import numpy as np
import pandas as pd
from miscc.config import cfg
from utils import get_imgs

import torch.utils.data as data

######################
###   Text Data    ###
######################
class TextDataset(data.Dataset):
    '''
    class about text data.

    Attributes:
        data_dir (str): parameter about data path.

        transform ( ): information to define of transform.
        target_transform ( ): information to define of target_transform.
        norm (transforms): transforms for narmalize.

        imsize (list): list of image size [64, 128, 256, ...].
    '''
    def __init__(self, data_dir, split='train', embedding_type='cnn-rnn',
                 base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        
        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.captions = self.load_all_captions()

        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_bbox(self):
        '''
        this is the function that making bbox.

        Returns:
            filename_bbox (dict{img_file_name: bbox}).
        '''
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                    delim_whitespace=True,
                                    header=None).astype(int)
        
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox
    
    def load_all_captions(self):
        '''
        This function works to load all captions.

        Returns:
            caption_dict (dict{file_name: list[str1, str2,..]}): \
                                    dictionary about file and captions of file.
        '''

        def load_captions(caption_name):
            '''
            This function works to load caption.
            
            Arguments:
                caption_name (str): path of caption.

            Returns:
                captions (list[str1, str2,...]): list of captions about 'caption_name'.
            '''
            cap_path = caption_name
            with open(cap_path, "r", encoding='utf8') as f:
                captions = f.read().split('\n')
            captions = [cap.replace("\ufffd\ufffd", " ")
                        for cap in captions if len(cap) > 0]
            return captions
        
        caption_dict = dict()
        for key in self.filenames:
            caption_name = '%s/text/%s.txt' % (self.data_dir, key)
            captions = load_captions(caption_name)
            caption_dict[key] = captions

        return caption_dict
    
    def load_embedding(self, data_dir:str, embedding_type:str)->np.array:
        '''
        This function works for loading embedding.
        embedding is already stored.
        
        Arguments:
            data_dir (str): path of data directory.
            embedding_type (str): define embedding type which you want.

        Returns:
            embeddings (ndarray): normaly, embedding size should 
                        [batch, number_of_sequences(captions), embedding_dim].
        '''
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)

        return embeddings
    
    def load_class_id(self, data_dir, total_num):
        '''
        This function works for making key index.
          If we already have class_info, we allocate prepaired class of files.
          But we haven't that information, then we just allocate linealy class_id.

        Arguments:
            data_dir (str): path of data directory.
            total_num (int): total number of keys.
        
        Returns:
            class_id (ndarray): ndarray about key index.
        '''
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        '''
        The function works for loading filenames.

        Arguments:
            data_dir (str): path of data directory.
        
        Returns:
            filenames (list): list of file names. File names are images name.
        '''
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames
    
    def prepair_training_pairs(self, index):
        '''
        The function that prepare to match the pairs of training set.

        Arguments:
            index (int): selected index of file.
        
        Returns:
            imgs (list): list of preprocessing images.
            wrong_imgs (list): list of preprocessing adversarial images.
            embedding (ndarray): embedding of one caption. [emb_dim].
            key (str): name of imgs.
        '''
        # key is file name
        key = self.filenames[index]

        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        
        # setup the wrong data on purpose randomly
        wrong_ix = random.randint(0, len(self.filenames) - 1)
        if(self.class_id[index] == self.class_id[wrong_ix]):
            wrong_ix = random.randint(0, len(self.filenames) - 1)
        wrong_key = self.filenames[wrong_ix]
        if self.bbox is not None:
            wrong_bbox = self.bbox[wrong_key]
        else:
            wrong_bbox = None
        wrong_img_name = '%s/images/%s.jpg' % (data_dir, wrong_key)
        wrong_imgs = get_imgs(wrong_img_name, self.imsize,
                              wrong_bbox, self.transform, normalize=self.norm)
        
        # select just one caption randomly
        embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        embedding = embeddings[embedding_ix, :]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)
        
        return imgs, wrong_imgs, embedding, key  # captions
    
    def prepair_test_pairs(self, index):
        '''
        The function that prepare to match the pairs of test set.

        Arguments:
            index (int): selected index of file.
        
        Returns:
            imgs (list): list of preprocessing images.
            embeddings (ndarray): embedding of one caption. [num_captions, emb_dim].
            key (str): name of imgs.
        '''
        # key is file name
        key = self.filenames[index]

        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        
        if self.target_transform is not None:
            embeddings = self.target_transform(embeddings)

        return imgs, embeddings, key  # captions
    
    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)