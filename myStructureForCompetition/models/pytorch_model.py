from torch import nn

from utils.config import CFG

class CustomModel(nn.Module):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model
        self.clf = nn.Sequential(
            nn.SiLU(),
            nn.LazyLinear(CFG.NUM_CLASSES),
        )

    def forward(self, x, label=None):
        x = self.model(x)
        logits = self.clf(x)
        loss = None
        if label is not None:
            loss = nn.CrossEntropyLoss()(logits, label)
        probs = nn.LogSoftmax(dim=-1)(logits)
        return probs, loss