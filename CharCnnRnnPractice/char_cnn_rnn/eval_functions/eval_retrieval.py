from collections import OrderedDict
import torch

def eval_retrieval(cls_feats_img, cls_feats_txt, cls_list, k_values=[1,5,10,50]):
    '''
    Retrieval evaluation (Average Precision).

    Arguments:
        cls_feats_img (list of torch.Tensor): list containing precomputed image
            features for each image, separated by class.
        cls_feats_txt (torch.Tensor): tensor containing precomputed (and
            averaged) textual features for each class.
        cls_list (list of string): list of class names.
        k_values (list, optional): list of k-values to use for evaluation.

    Returns:
        map_at_k (OrderedDict): dictionary whose keys are the k_values and the
            values are the mean Average Precision (mAP) for all classes.
        cls_stats (OrderedDict): dictionary whose keys are class names and each
            entry is a dictionary whose keys are the k_values and the values
            are the Average Precision (AP) per class.
    '''
    total_num_cls = cls_feats_txt.size(0)
    total_num_img = sum([feats.size(0) for feats in cls_feats_img])
    scores = torch.zeros(total_num_cls, total_num_img)
    matches = torch.zeros(total_num_cls, total_num_img)

    for i, cls in enumerate(cls_list):
        start_id = 0
        for j, feats_img in enumerate(cls_feats_img):
            # feats_img is [num_feats_img, 1024]
            # cls_feats_txt is [num_class, 1024]
            # so, feats_img.size is 'num_feats_img'
            end_id = start_id + feats_img.size(0)
            # torch.matmul size is [num_feats_img, ]
            scores[i, start_id:end_id] = torch.matmul(feats_img, cls_feats_txt[i])
            # if i==j, it means class is same, then matches should be 1(highest score)
            if i == j: matches[i, start_id:end_id] = 1
            start_id = start_id + feats_img.size(0)
    
    for i, s in enumerate(scores): # for each class
        # inds is the index that before sorted
        _, inds = torch.sort(s, descending=True)
        # it re_order 'matches' with the rule of 'descendin scores
        matches[i] = matches[i, inds]

        #This allows you to sort the images in the order 
        #   in which the scores for each class are highest, 
        # and to update the matches in that order. 
        # This is typically used to calculate 
        #   precision-reproducibility curves, ROC curves, and so on.

    map_at_k = OrderedDict()
    for k in k_values: # for mAP
        map_at_k[k] = torch.mean(matches[:, 0:k]).item()

    cls_stats = OrderedDict()
    for i, cls in enumerate(cls_list):
        ap_at_k = OrderedDict()
        for k in k_values:
            ap_at_k[k] = torch.mean(matches[i, 0:k]).item()
        cls_stats[cls] = ap_at_k

    return map_at_k, cls_stats