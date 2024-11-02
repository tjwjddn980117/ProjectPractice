import cv2
import os
import numpy as np
from CI import collective_insight_score
from GED import generalized_energy_distance
from hm_IoU import hungarian_matching_iou
from NCC import variance_ncc_dist
from CPI import continuous_performance_index
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

def load_from_folder_with_compare_the_folder(labels_path: str, pred_paths: str) -> list[list[str]]:
    # 1. 디렉토리 내 파일 목록 불러오기
    labels_files = sorted(os.listdir(labels_path))
    preds_files = sorted(os.listdir(pred_paths))

    # 2. preds_files에 있는 하위 폴더 이름만 labels_files에 남기기
    preds_set = set(os.path.basename(p) for p in preds_files)  # preds_files에서 하위 폴더 이름만 추출
    labels_files = [lf for lf in labels_files if os.path.basename(lf) in preds_set]

    paths = []
    # 3. 필터링된 labels_files를 사용하여 라벨과 예측 파일 매칭
    for label_file in labels_files:
        l_p = os.path.join(labels_path, label_file)
        p_p = os.path.join(pred_paths, label_file)

        # 라벨 파일과 예측 파일 로드
        l = sorted([i for i in os.listdir(l_p) if 'label' in i])
        p = sorted(os.listdir(p_p))
        
        paths.append([[os.path.join(l_p, ll) for ll in l], [os.path.join(p_p, pp) for pp in p]])
    
    return paths
    
def calc_matric(labels_path: str, preds_paths: str, reverse: bool=False) -> dict[str, float]:
    
    # eval_paths = load_from_folder(labels_path, preds_paths)
    eval_paths = load_from_folder_with_compare_the_folder(labels_path, preds_paths)
    
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

def calc_matric_with_var(labels_path: str, preds_paths: str, reverse: bool=False) -> dict[str, float]:
    # Initialize lists to hold individual scores

    eval_paths = load_from_folder_with_compare_the_folder(labels_path, preds_paths)
    total_CI_score = 0
    total_ged = 0
    total_hm_iou = 0
    total_ncc = 0
    total_CPI = 0
    ci_scores = []
    ged_scores = []
    hm_iou_scores = []
    ncc_scores = []
    CPI_scores = []

    if 'image' in preds_paths:
        load_pred = load_images
    else:
        load_pred = load_tensors
    num_images = len(eval_paths)

    for labels, preds in tqdm(eval_paths):
        labels = load_images(labels)
        preds = load_pred(preds, reverse)

        ci = collective_insight_score(labels, preds)
        if ci > 0 and ci < 2:
            total_CI_score += ci
            ci_scores.append(ci)  # Append to the list for variance calculation

        ged = generalized_energy_distance(labels, preds)
        total_ged += ged
        ged_scores.append(ged)  # Append to the list for variance calculation

        hm_iou = hungarian_matching_iou(labels, preds)
        total_hm_iou += hm_iou
        hm_iou_scores.append(hm_iou)  # Append to the list for variance calculation

        ncc = variance_ncc_dist(labels, preds)
        total_ncc += ncc
        ncc_scores.append(ncc)  # Append to the list for variance calculation

        cpi = continuous_performance_index(labels, preds)
        total_CPI += cpi
        CPI_scores.append(cpi)

    # Calculate means
    total_CI_score /= num_images
    total_ged /= num_images
    total_hm_iou /= num_images
    total_ncc /= num_images
    total_CPI /= num_images

    # Calculate variances
    var_CI_score = np.var(ci_scores) if ci_scores else 0
    var_ged = np.var(ged_scores) if ged_scores else 0
    var_hm_iou = np.var(hm_iou_scores) if hm_iou_scores else 0
    var_ncc = np.var(ncc_scores) if ncc_scores else 0
    var_CPI = np.var(CPI_scores) if CPI_scores else 0


    # Example output format for the experiment
    print(f"CI Score: {total_CI_score} ± {np.sqrt(var_CI_score)}")
    print(f"GED: {total_ged} ± {np.sqrt(var_ged)}")
    print(f"hm iou: {total_hm_iou} ± {np.sqrt(var_hm_iou)}")
    print(f"NCC: {total_ncc} ± {np.sqrt(var_ncc)}")
    print(f"CPI: {total_CPI} ± {np.sqrt(var_CPI)}")


label_path = 'C:/Users/Seo/Downloads/our_MS-MRI_dataset_split/testing'

#pred_path = 'C:/Users/Seo/Downloads/MS_MRI_tensors/cFlow_MSMRI_tensor'
#print()
#print("cFlow_MSMRI")
##print(calc_matric(label_path, pred_path, reverse=False))
#calc_matric_with_var(label_path, pred_path, reverse=False)
#
#pred_path = 'C:/Users/Seo/Downloads/MS_MRI_tensors/MoSE_MSMRI_tensor'
#print()
#print("MoSE_MSMRI")
##print(calc_matric(label_path, pred_path, reverse=False))
#calc_matric_with_var(label_path, pred_path, reverse=False)
#
#pred_path = 'C:/Users/Seo/Downloads/MS_MRI_tensors/CCDM_MSMRI_tensor'
#print()
#print("CCDM_MSMRI")
#calc_matric_with_var(label_path, pred_path, reverse=False)
#
#pred_path = 'C:/Users/Seo/Downloads/MS_MRI_tensors/CIMD_MSMRI_tensor'
#print()
#print("CIMD_MSMRI")
#calc_matric_with_var(label_path, pred_path, reverse=False)
#
#pred_path = 'C:/Users/Seo/Downloads/MS_MRI_tensors/MoDiff_MSMRI_tensor'
#print()
#print("MoDiff_MSMRI")
#calc_matric_with_var(label_path, pred_path, reverse=False)
# 
# pred_path = 'C:/Users/Seo/Downloads/MS_MRI_tensors/ProbUnet_MSMRI_tensor'
# print()
# print("ProbUnet_MSMRI")
# calc_matric_with_var(label_path, pred_path, reverse=False)

# pred_path = 'C:/Users/Seo/Downloads/MoSE_MSMRI_results_32_sample/tensor'
# print()
# print("MoSE_MSMRI_32_sample")
# calc_matric_with_var(label_path, pred_path, reverse=False)
# 
# pred_path = 'C:/Users/Seo/Downloads/cFlow_MSMRI_results_32_sample/tensor'
# print()
# print("cFlow_MSMRI_32_sample")
# calc_matric_with_var(label_path, pred_path, reverse=False)

# pred_path = 'C:/Users/Seo/Downloads/MoSE_LIDC_results_32_sample/tensor'
# print()
# print("MoSE LIDC 32 sample")
# calc_matric_with_var(label_path, pred_path, reverse=False)

pred_path = 'Downloads/CIMD_MSMRI_tensor'
print()
print("CIMD MSMRI final")
calc_matric_with_var(label_path, pred_path, reverse=False)