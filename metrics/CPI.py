import cv2
import os
import numpy as np

def calc_D_c(m1: np.ndarray, m2: np.ndarray) -> float:
    intersection = np.sum(np.multiply(m1, m2))
    union = np.sum(m1**2) + np.sum(m2**2) 
    
    if union == 0:
        return 1e-6
    
    return 2 * intersection / union

def calc_r(y_p: np.ndarray, y_hat_p: np.ndarray) -> float:
    y_mean = np.mean(y_p)
    y_hat_mean = np.mean(y_hat_p)
    
    numerator = np.sum(np.multiply((y_p - y_mean), (y_hat_p - y_hat_mean)))
    denominator = (((np.sum(y_p - y_mean)**2)**0.5) * ((np.sum(y_hat_p - y_hat_mean)**2)**0.5))
    
    if denominator == 0:
        return 1e-6
    
    return numerator / denominator

def calc_r_norm(gt: np.ndarray, pred: np.ndarray) -> float:
    r = calc_r(gt, pred)
    return (r + 1) / 2

def calc_V(preds: list[np.ndarray]) -> float:
    N = len(preds)

    if N < 2:  # preds의 길이가 2보다 작으면 0 반환
        return 0
    
    total_D_c = 0
    
    for i in range(len(preds)-1):
        for j in range(i+1, len(preds)):
            total_D_c = calc_D_c(preds[i], preds[j])
    
    total_D_c = 2 * total_D_c / (N * (N - 1))
    
    return 1 - total_D_c

def continuous_performance_index(gt: list[np.ndarray], pred: list[np.ndarray]) -> float:
    N = len(gt)
    M = len(pred)
    
    D_c = 0
    r_norm = 0
    
    for y_p in gt:
        for y_hat_p in pred:
            D_c += calc_D_c(y_p, y_hat_p)
            r_norm += calc_r_norm(y_p, y_hat_p)
    
    D_c /= (N * M)
    r_norm /= (N * M)
    
    V = calc_V(pred)

    # 분모 계산
    denominator = (D_c * r_norm + D_c * V + r_norm * V)
    
    if denominator == 0:  # 분모가 0이 되는 경우
        return 0
    
    return (3 * D_c * r_norm * V) / denominator
