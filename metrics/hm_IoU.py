
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import jaccard_score
import numpy as np
import cv2

def iou_score(gt, pred):
    """IoU(Intersection over Union)를 계산하는 함수"""
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    if np.sum(union) == 0:
        return 1.0  # 공집합일 때 IoU는 1
    return np.sum(intersection) / np.sum(union)

# def hungarian_matching_iou(ground_truth, predictions):
#     # ground_truth: 실제 세그먼트 (2D numpy array)
#     # predictions: 예측 세그먼트 (list of 2D numpy arrays)

#     num_predictions = len(predictions)
#     cost_matrix = np.zeros((num_predictions, num_predictions))

#     # IoU를 기반으로 cost matrix 작성
#     for i, gt in enumerate(ground_truth):
#         for j, pred in enumerate(predictions):
#             intersection = np.logical_and(gt, pred)
#             union = np.logical_or(gt, pred)
#             iou = np.sum(intersection) / np.sum(union)
#             cost_matrix[i, j] = 1 - iou  # 비용 행렬이므로 1 - IoU를 사용

#     # 헝가리안 알고리즘을 사용해 최적의 매칭을 찾음
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)

#     matched_iou = []
#     for i, j in zip(row_ind, col_ind):
#         intersection = np.logical_and(ground_truth[i], predictions[j])
#         union = np.logical_or(ground_truth[i], predictions[j])
#         iou = np.sum(intersection) / np.sum(union)
#         matched_iou.append(iou)

#     return np.mean(matched_iou)  # 최적 매칭의 IoU 평균

def hungarian_matching_iou(ground_truth: list[np.ndarray[float]], predictions: list[np.ndarray[float]]):
    matched_iou = []
    matched_pred_index = set()
    
    for gt in ground_truth:
        max_iou = []
        for i, pred in enumerate(predictions):
            max_iou.append((iou_score(gt, pred), i))
        
        max_iou.sort(key=lambda x: (-x[0], x[1]))
        
        for iou, i in max_iou:
            if i not in matched_pred_index:
                matched_iou.append(iou)
                matched_pred_index.add(i)
                break
               
    return np.mean(matched_iou)

# from load import load_images
# pred_path = ['training/0/label0_.jpg', 'training/0/label1_.jpg']
# gt_path = ['training/0/label2_.jpg', 'training/0/label3_.jpg']

# pred_images = load_images(pred_path)
# gt_images = load_images(gt_path)

# print(hungarian_matching_iou(gt_images, pred_images))
