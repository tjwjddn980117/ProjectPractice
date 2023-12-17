from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from ..miscc.config import cfg

import torch.utils.data as data
from PIL import Image
import os
import os.path
import six
import string
import sys
import torch

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    ''' collect all image files what we have. like \'.jpg\', \'.PNG\', ... '''
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_imgs(img_path, imsize, bbox=None, transform=None, normalize=None):
    ''' 
    This function loads a given image, optionally cuts out a specific area, 
    applies a transformation, readjusts to multiple sizes, 
    and returns the normalized results to the list.

    Arguments:
        img_path (str) : the path of image dir
        imsize ( ) : it should width, height
        bbox ( ) : boundary box. standard for cutting out a picture
        transform ( ) : the parameter that standard transform
        normalize ( ) : the parameter that standard normalize
    
    Returns:
        list [] : list of resized images
    '''
    img = Image.open(img_path).convert('RGB')
    width, height = img.size

    # cropping the image
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75) # canculate maximum range
        center_x = int((2 * bbox[0] + bbox[2]) / 2) # define center_x
        center_y = int((2 * bbox[1] + bbox[3]) / 2) # define center_y
        y1 = np.maximum(0, center_y - r) # define y1's point
        y2 = np.minimum(height, center_y + r) # define y2's point
        x1 = np.maximum(0, center_x - r) # define x1's point
        x2 = np.minimum(width, center_x + r) # define x2's point
        img = img.crop([x1, y1, x2, y2]) # cropping
        # this function is just cropping with pictures

    if transform is not None:
        img = transform(img)
    
    ret = []
    for i in range(cfg.TREE.BRANCH_NUM):
        # resizing image for batch_size
        if i < (cfg.TREE.BRANCH_NUM - 1):
            re_img = transforms.Scale(imsize[i])(img)
        else:
            # last image dosen't transform scale last batch
            re_img = img
        
        # append all re_sizing image with normalize
        ret.append(normalize(re_img))
    
    return ret

class ImageFolder(data.Dataset):
    def __init__(self, root, split_dir='train', custom_classes=None,
                 base_size=64, transform=None, target_transform=None):
        root = os.path.join(root, split_dir)
        classes, class_to_idx = self.find_classes(root, custom_classes)


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
            images (list): 
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
                        images.append(item)
        print('The number of images: ', len(images))
        return images