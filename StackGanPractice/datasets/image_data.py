from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.utils.data as data
import torchvision.transforms as transforms
import os
import os.path
from ..miscc.config import cfg
from utils import *

import torch.utils.data as data
import os
import os.path

######################
###   Image Data   ###
######################
class ImageFolder(data.Dataset):
    '''
    class about image data.

    Attributes:
        root (str): this is the root path of image data.
        imgs (list[(path, class_index)]): this is the list of image data tuple (path, class_index).
        classes (list[path]): this is the list of each class path.
        num_classes (int): this is the number of classes.
        class_to_idx (dict{class_path: int}): index information with class_path and index

        transform ( ): information to define of transform
        target_transform ( ): information to define of target_transform
        norm (transforms): transforms for narmalize

        imsize (list): list of image size [64, 128, 256, ...]
    '''
    def __init__(self, root, split_dir='train', custom_classes=None,
                 base_size=64, transform=None, target_transform=None):
        root = os.path.join(root, split_dir)
        classes, class_to_idx = self.find_classes(root, custom_classes)
        imgs = self.make_dataset(classes, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.num_classes = len(classes)
        self.class_to_idx = class_to_idx

        self.transform = transform
        self.target_transform = target_transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # imsize should be [64, 128, 256, ...]
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        print('num_classes', self.num_classes)

    def find_classes(self, dir:str, custom_classes):
        ''' 
        the function that find classes 
        Arguments:
            dir (str): the root of directory
            custom_classes (list): The list of classes you want to find. \
                                    If this value is None, find all classes.

        Returns:
            clsses (list): classes path list which selected sorted
            class_to_idx (dict): {class_path: index} \
                                Index each class in the classes list \
                                and save it in the form of a dictionary.
        '''
        classes = []

        for sub_dir in os.listdir(dir):
            if os.path.isdir: # only directory can open
                if custom_classes is None or sub_dir in custom_classes:
                    classes.append(os.path.join(dir, sub_dir))
        print('Valid classes: ', len(classes), classes)

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def make_dataset(self, classes:list, class_to_idx:dict):
        '''
        the function that make dataset
        Arguments:
            classes (list): classes path list wich selected sorted
            class_to_idx (dict): {class_path: index} \
                                Index each class in the classes list\
                                and save it in the form of a dictionary.
        Returns:
            images (list): image files [(path, class_index)...]
        '''
        images = []
        for class_dir in classes:
            # root is path
            # fnames is file names
            for root, _, file_names in sorted(os.walk(class_dir)):
                for file in file_names:
                    if is_image_file(file):
                        path = os.path.join(root, file)
                        item = (path, class_to_idx[class_dir])
                        # save the tuple
                        images.append(item)
        print('The number of images: ', len(images))
        # put image files [(path, class_index)...]
        return images
    
    def __getitem__(self, index)->list:
        '''
        call the data
        Arguments:
            index (int): index of image. the image is tuple of (path, class_idx)
        Returns:
            imgs_list (list[Image]): list of resized images data type is (Imgae)
        '''
        # imgs tuple of (path, class_idx)
        path, target = self.imgs[index]
        imgs_list = get_imgs(path, self.imsize,
                             transform=self.transform,
                             normalize=self.norm)
        return imgs_list
    
    def __len__(self):
        return len(self.imgs)
    