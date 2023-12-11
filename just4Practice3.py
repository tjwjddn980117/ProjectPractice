import torch

def str_to_labelvec(string, max_str_len):
    string = string.lower()
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    alpha_to_num = {k:v+1 for k,v in zip(alphabet, range(len(alphabet)))}
    labels = torch.zeros(max_str_len).long()
    max_i = min(max_str_len, len(string)) # compare lenght of maximum VS string. force the size to max
    for i in range(max_i):
        labels[i] = alpha_to_num.get(string[i], alpha_to_num[' '])
    
    return labels

labels = str_to_labelvec("abd",4)
#print(labels.size())
#print(labels)
labels = torch.LongTensor(labels).unsqueeze(1)
#print(labels.size())
#print(labels)
one_hot = torch.zeros(labels.size(0), 8).scatter_(1, labels, 1.)
#print(one_hot.size())
#print(one_hot)
one_hot = one_hot[:, 1:]
# [max_str_len, emb_size] to [emb_size, max_str_len]
print(one_hot)
one_hot = one_hot.permute(1,0)

print(one_hot)