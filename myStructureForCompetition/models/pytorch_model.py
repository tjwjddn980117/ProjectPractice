import torch
from torch import nn

from utils.config import CFG
from utils.check_missing_value import zero_filtering, check_nan, nan_filtering

class CustomModel(nn.Module):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model
        self.clf = nn.Sequential(
            nn.SiLU(),
            nn.LazyLinear(CFG.NUM_CLASSES*3),
            nn.BatchNorm1d(CFG.NUM_CLASSES*3),
            nn.Linear(CFG.NUM_CLASSES*3, CFG.NUM_CLASSES)
        )

    def forward(self, x, label=None):
        x = self.model(x)
        #if check_nan(x):
        #    print('\n there has some Nan data in the input data. ')
        #    print(x)
        #    x = torch.nan_to_num(x, nan=1e-10)
        #    print('\n there has some Nan data in after num_to_num. ')
        #    print(x)
        x = torch.nan_to_num(x, nan=0.0)
        #if check_nan(x):
        #    print('\n there has some Nan data in after num_to_num. ')
        #    print(x)
        logits = self.clf(x)

        min_threshold = 1e-10  # 필요에 따라 임계값을 조정
        #logits = torch.clamp(logits, min=min_threshold)

        loss = None
        if label is not None:
            loss = nn.CrossEntropyLoss()(logits, label)
        
        #probs = zero_filtering(probs)
        # if check_nan(logits):
        #     print('\n there has some Nan data in the clf. ')
        #     print(logits)

        probs = nn.functional.softmax(logits, dim=-1)

        # if check_nan(probs):
        #     print('\n there has some Nan data after softmax. ')
        #     print(probs)
        probs = torch.clamp(probs, min=min_threshold)

        # if check_nan(probs):
        #     print('\n there has some Nan data after clapm. ')
        #     print(probs)

        #probs = zero_filtering(probs)
        probs = torch.log(probs)

        # if check_nan(probs):
        #     print('\n there has some Nan data after logs. ')
        #     print(probs)
        #   return probs, loss, True
        
        return probs, loss, False