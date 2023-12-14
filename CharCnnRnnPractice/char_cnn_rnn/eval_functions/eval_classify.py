from collections import OrderedDict
import torch

def eval_classify(cls_feats_img, cls_feats_txt, cls_list):
    '''
    Classification evaluation.

    Arguments:
        cls_feats_img (list of torch.Tensor): list containing precomputed image
            features for each image, separated by class.
        cls_feats_txt (torch.Tensor): tensor containing precomputed (and
            averaged) textual features for each class.
        cls_list (list of string): list of class names.

    Returns:
        avg_acc (float): percentage of correct classifications for all classes.
        cls_stats (OrderedDict): dictionary whose keys are class names and each
            entry is a dictionary containing the 'total' of images for the
            class and the number of 'correct' classifications.
    '''

    cls_stats = OrderedDict()
    # for classes
    for i, cls in enumerate(cls_list):
        feats_img = cls_feats_img[i]
        # [num_feats_img, 1024] matmul [1024, num_class]
        # scores should be [num_feats_img, num_class]
        scores = torch.matmul(feats_img, cls_feats_txt.t())
        # check the highest score index [num_feats_img, ]
        max_ids = torch.argmax(scores, dim=1).to('cpu')
        # ground truths will filled with classes(index) [num_feats_img, ]
        ground_truths = torch.LongTensor(scores.size(0)).fill_(i)
        num_correct = (max_ids == ground_truths).sum().item()
        cls_stats[cls] = {'correct': num_correct, 'total': ground_truths.size(0)}

    total = sum([stats['total'] for _, stats in cls_stats.items()])
    total_correct = sum([stats['correct'] for _, stats in cls_stats.items()])
    avg_acc = total_correct / total

    # avg_acc (float)
    # cls_stats (OrderedDict)
    return avg_acc, cls_stats