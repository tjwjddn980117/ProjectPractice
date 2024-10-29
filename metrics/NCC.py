import cv2
import os
import numpy as np

# 참고
# https://arxiv.org/pdf/1906.04045
# https://github.com/baumgach/PHiSeg-code/blob/master/utils.py#L103

def ncc(a,v, zero_norm=True):

    a = a.flatten()
    v = v.flatten()

    if zero_norm:
        # 평균 0, 표준편차 1로 정규화
        a_mean = np.mean(a)
        v_mean = np.mean(v)
        a_std = np.std(a)
        v_std = np.std(v)
        
        # 표준 편차가 0인 경우 처리
        if a_std == 0:
            a_std = 1e-9
        if v_std == 0:
            v_std = 1e-9

        a = (a - a_mean) / (a_std * len(a))
        v = (v - v_mean) / v_std
    else:
        a_std = np.std(a)
        v_std = np.std(v)

        if a_std == 0:
            a_std = 1e-9
        if v_std == 0:
            v_std = 1e-9

        a = a / (a_std * len(a))
        v = v / v_std

    return np.correlate(a,v)

def variance_ncc_dist(gt_arr, sample_arr):

    def pixel_wise_xent(m_samp, m_gt, eps=1e-8):


        log_samples = np.log(m_samp + eps)

        return -1.0*np.sum(m_gt*log_samples, axis=-1)

    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y
    :return: 
    """
    gt_arr = np.array(gt_arr)
    sample_arr = np.array(sample_arr)

    mean_seg = np.mean(sample_arr, axis=0)

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    sX = sample_arr.shape[1]
    sY = sample_arr.shape[2]

    E_ss_arr = np.zeros((N,sX,sY))
    for i in range(N):
        E_ss_arr[i,...] = pixel_wise_xent(sample_arr[i,...], mean_seg)
        # print('pixel wise xent')
        # plt.imshow( E_ss_arr[i,...])
        # plt.show()

    E_ss = np.mean(E_ss_arr, axis=0)

    E_sy_arr = np.zeros((M,N, sX, sY))
    for j in range(M):
        for i in range(N):
            E_sy_arr[j,i, ...] = pixel_wise_xent(sample_arr[i,...], gt_arr[j,...])

    E_sy = np.mean(E_sy_arr, axis=1)

    ncc_list = []
    for j in range(M):

        ncc_list.append(ncc(E_ss, E_sy[j,...]))

    return (1/M)*sum(ncc_list)


# from load import load_images
# # S_NCC 계산
# pred_path = ['training/0/label0_.jpg', 'training/0/label1_.jpg']
# gt_path = ['training/0/label2_.jpg', 'training/0/label3_.jpg']

# pred_images = load_images(pred_path)
# gt_images = load_images(gt_path)

# print(s_ncc(gt_images[0], pred_images)) # 왜인지 모르겠는데, nan뜸 
# print(ncc(gt_images, pred_images))
# ncc0 = ncc_segmentation(gt_images[0], pred_images[0])
# ncc1 = ncc_segmentation(gt_images[1], pred_images[1])
# print(ncc0) # gt 1개, pred 1개
# print(ncc1) # gt 1개, pred 1개

# print((ncc0+ncc1)/2)