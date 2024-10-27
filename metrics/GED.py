
import numpy as np
import cv2

# 참고
# https://github.com/aimansnigdha/Ambiguous-Medical-Image-Segmentation-using-Diffusion-Models/blob/main/scripts/.ipynb_checkpoints/segmentation_sampleqaun-checkpoint.py

def iou_score(pred, gt):
    """IoU(Intersection over Union)를 계산하는 함수"""
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    if np.sum(union) == 0:
        return 1.0  # 공집합일 때 IoU는 1
    return np.sum(intersection) / np.sum(union)

def distance(pred, gt):
    """d(x, y) = 1 - IoU(x, y)로 두 세그먼트 간의 거리를 계산"""
    return 1 - iou_score(pred, gt)

def generalized_energy_distance(ground_truths, predictions):
    """
    Generalized Energy Distance (GED) 계산 함수
    
    Parameters:
    - ground_truths: 실제 세그먼테이션 리스트 (list of numpy arrays)
    - predictions: 예측 세그먼테이션 리스트 (list of numpy arrays)
    
    Returns:
    - GED 값 (float)
    """
    num_predictions = len(predictions)
    num_ground_truths = len(ground_truths)

    # 2 * E[d(S, Y)] 계산
    term1 = 0
    for pred in predictions:
        for gt in ground_truths:
            term1 += distance(pred, gt)
    term1 /= (num_predictions * num_ground_truths)

    # E[d(S, S')] 계산
    term2 = 0
    for i in range(num_predictions):
        for j in range(i + 1, num_predictions):
            term2 += distance(predictions[i], predictions[j])
    if num_predictions > 1:
        term2 /= (num_predictions * (num_predictions - 1) / 2)

    # E[d(Y, Y')] 계산
    term3 = 0
    for i in range(num_ground_truths):
        for j in range(i + 1, num_ground_truths):
            term3 += distance(ground_truths[i], ground_truths[j])
    if num_ground_truths > 1:
        term3 /= (num_ground_truths * (num_ground_truths - 1) / 2)

    # GED 계산
    # print(term1, term2, term3)
    ged = 2 * term1 - term2 - term3
    return ged

# from load import load_images
# pred_path = ['training/0/label0_.jpg', 'training/0/label1_.jpg']
# gt_path = ['training/0/label2_.jpg', 'training/0/label3_.jpg']

# pred_images = load_images(pred_path)
# gt_images = load_images(gt_path)

# print(generalized_energy_distance(gt_images, pred_images))

