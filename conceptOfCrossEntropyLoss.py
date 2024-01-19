import torch
def NLLLoss(logs, targets):
    out = torch.zeros_like(targets, dtype=torch.float)
    for i in range(len(targets)):
        out[i] = logs[i][targets[i]]
        print(logs[i][targets[i]])
    return -out.sum()/len(out)

output = torch.Tensor(
    [
        [0.8982, 0.805, 0.6393, 0.9983, 0.5731, 0.0469, 0.556, 0.1476, 0.8404, 0.5544],
        [0.9457, 0.0195, 0.9846, 0.3231, 0.1605, 0.3143, 0.9508, 0.2762, 0.7276, 0.4332]
    ]
)
y = torch.LongTensor([1, 5])

# Case 2
log_softmax = torch.nn.LogSoftmax(dim=1)
x_log = log_softmax(output)
print(x_log)
print(NLLLoss(x_log, y)) # tensor(2.1438)

