import cv2
import random
import numpy as np

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None , flag_mixup = False):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        self.flag_mixup = flag_mixup
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        if self.label_list is not None:
            label = torch.zeros(len(le.classes_))
            label[self.label_list[index]] = 1.
            
        # 기존 image data
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.flag_mixup :
            # mixup에 사용될 data 선택
            mixup_label = torch.zeros(len(le.classes_))

            while True:
              mixup_idx = random.randint(0, len(self.img_path_list)-1) # 전체 데이터 중 아무거나 선택 / 중복되는 클래스가 선택될 수 있음
              if self.label_list[mixup_idx] != self.label_list[index]: # 같은 카테고리 방지
                mixup_label[self.label_list[mixup_idx]] = 1.
                break
        
            # mix할 이미지
            mixup_image = cv2.imread(self.img_path_list[mixup_idx])
            if self.transforms is not None:
                mixup_image = self.transforms(image = mixup_image)["image"]

            # Select a random number from the given beta distribution
            # Mixup the images accordingly
            alpha = 0.4
            lam = np.random.beta(alpha, alpha)
            image = lam * image + (1 - lam) * mixup_image
            label = lam * label + (1 - lam) * mixup_label

        # label one-hot으로 생성
        if self.label_list is not None:
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)