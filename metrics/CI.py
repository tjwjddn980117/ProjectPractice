import cv2
import os
import numpy as np

# 참고
# https://arxiv.org/pdf/2304.04745

def combined_sensitivity(ground_truth, predictions):
    # Y_c: ground truth의 합집합
    # Ŷ_c: predictions의 합집합
    Y_c = np.logical_or.reduce(ground_truth)
    Y_hat_c = np.logical_or.reduce(predictions)

    # TP (True Positive): Ground truth와 예측의 교집합
    TP = np.sum(np.logical_and(Y_c, Y_hat_c))
    # FN (False Negative): Ground truth는 있지만 예측은 없는 부분
    FN = np.sum(np.logical_and(Y_c, np.logical_not(Y_hat_c)))

    if np.sum(np.logical_or(Y_c, Y_hat_c)) == 0:  # Y_c와 Y_hat_c가 둘 다 공집합일 때
        return 1.0  # S_c = 1
    return TP / (TP + FN)

def dice_score(Y_hat, Y):
    intersection = np.sum(np.logical_and(Y_hat, Y))
    if np.sum(Y_hat) + np.sum(Y) == 0:
        return 1.0  # 둘 다 공집합일 때
    return 2 * intersection / (np.sum(Y_hat) + np.sum(Y))

def max_dice_score(ground_truth, predictions):
    D_i = []
    for i in range(len(ground_truth)):
        dice_scores = [dice_score(pred, ground_truth[i]) for pred in predictions]
        D_i.append(np.max(dice_scores))  # 각 ground truth에 대한 최대 Dice score
    return np.mean(D_i)  # D_max는 각 ground truth에 대한 최대값들의 평균

def cal_variance(img1, img2):
    diff = img1 - img2
    var = np.var(diff)
    return var

def diversity_agreement(ground_truths, predictions):
    # V_min, V_max는 ground truth와 예측 간의 분산 차이
    
    V_Y = []
    V_hat = []
    
    for i in range(len(ground_truths)-1):
        for j in range(i+1, len(ground_truths)):
            V_Y.append(cal_variance(ground_truths[i], ground_truths[j]))
    
    for i in range(len(predictions)-1):
        for j in range(i+1, len(predictions)):
            V_hat.append(cal_variance(predictions[i], predictions[j]))
    
    ΔV_min = abs(np.min(V_Y) - np.min(V_hat))
    ΔV_max = abs(np.max(V_Y) - np.max(V_hat))

    D_a = 1 - ((ΔV_max + ΔV_min) / 2)
    return D_a

def collective_insight_score(ground_truth, predictions):
    S_c = combined_sensitivity(ground_truth, predictions)
    D_max = max_dice_score(ground_truth, predictions)
    D_a = diversity_agreement(ground_truth, predictions)
    
    CI = (3 * S_c * D_max * D_a) / (S_c + D_max + D_a)
    return CI

# from load import load_images
# pred_path = ['training/0/label0_.jpg', 'training/0/label1_.jpg']
# gt_path = ['training/0/label2_.jpg', 'training/0/label3_.jpg']

# pred_images = load_images(pred_path)
# gt_images = load_images(gt_path)

# print(collective_insight_score(pred_images, gt_images))