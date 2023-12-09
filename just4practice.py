import torch
from torch import nn


def str_to_labelvec(string, max_str_len):
    string = string.lower()
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    alpha_to_num = {k:v+1 for k,v in zip(alphabet, range(len(alphabet)))}
    labels = torch.zeros(max_str_len).long()
    max_i = min(max_str_len, len(string)) # compare lenght of maximum VS string. force the size to max
    for i in range(max_i):
        labels[i] = alpha_to_num.get(string[i], alpha_to_num[' '])
    
    return labels

print(str_to_labelvec("abc!",10))