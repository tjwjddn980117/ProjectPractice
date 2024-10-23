import cv2
import os
import numpy as np
from CI import collective_insight_score
from GED import generalized_energy_distance
from hm_IoU import hungarian_matching_iou
from NCC import variance_ncc_dist
import torch.nn as nn
import torch
from tqdm import tqdm


def load_images(image_paths: str, reverse: bool=False) -> list[np.ndarray]:
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 이미지를 그레이스케일로 불러옴
        img = img / 255.0
        img = np.where(img >= 0.5, 1, 0)
        if reverse:
            img = 1 - img
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: {path} could not be loaded.")
    return images

def load_tensors(image_paths: str, revers: bool=False) -> list[np.ndarray]:
    images = []
    sigmoid = nn.Sigmoid()
    
    for path in image_paths:
        img = torch.load(path, map_location=torch.device('cpu'))
        
        if len(img.shape) == 4:
            img = torch.squeeze(img, 0)
        
        # img = sigmoid(img)
        
        img = (img >= 0.5).float()
        
        if revers:
            img = 1 - img
        if img is not None:
            images.append(img.numpy())
        else:
            print(f"Warning: {path} could not be loaded.")
    
    return images

def load_from_folder(labels_path: str, pred_paths: str) -> list[list[str], list[str]]:
    labels_files = sorted(os.listdir(labels_path))
    preds_files = sorted(os.listdir(pred_paths))
    assert len(labels_files)==len(preds_files), "not the same number"
    
    paths = []
    for i in range(len(labels_files)):
        l_p = os.path.join(labels_path, labels_files[i])
        p_p = os.path.join(pred_paths, preds_files[i])
        
        l = sorted([i for i in os.listdir(l_p) if 'label' in i])
        p = sorted(os.listdir(p_p))
        
        paths.append([[os.path.join(l_p, ll) for ll in l], [os.path.join(p_p, pp) for pp in p]])
    
    return paths
    
def calc_matric(labels_path: str, preds_paths: str, reverse: bool=False) -> dict[str, float]:
    
    eval_paths = load_from_folder(labels_path, preds_paths)
    
    total_CI_score = 0
    total_ged = 0
    total_hm_iou = 0
    total_ncc = 0
    
    num_images = len(eval_paths)
    
    if 'image' in preds_paths:
        load_pred = load_images
    else:
        load_pred = load_tensors
    
    for labels, preds in tqdm(eval_paths):
        labels = load_images(labels)
        preds = load_pred(preds, reverse)
        ci = collective_insight_score(labels, preds)
        if ci > 0 and ci < 2:
            total_CI_score += collective_insight_score(labels, preds)
        total_ged += generalized_energy_distance(labels, preds)
        total_hm_iou += hungarian_matching_iou(labels, preds)
        total_ncc += variance_ncc_dist(labels, preds)

    total_CI_score /= num_images
    total_ged /= num_images
    total_hm_iou /= num_images
    total_ncc /= num_images
        
    return dict(mCI_score=total_CI_score,
                mGED=total_ged,
                mHM_iou=total_hm_iou,
                ncc=total_ncc[0])   

label_path = 'C:/Users/Seo/Downloads/our_LIDC-IDRI_dataset/testing'
pred_path = 'C:/Users/Seo/Downloads/cFlow-master/Results_3/tensor'

print("cFlow_LIDC")
print(calc_matric(label_path, pred_path, reverse=False))

pred_path = 'C:/Users/Seo/Downloads/MoSE_lidc_results/tensor'
print()
print("MoSE_LIDC")
print(calc_matric(label_path, pred_path, reverse=False))