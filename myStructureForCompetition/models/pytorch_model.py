from torch import nn
import pytorch_lightning as L

from sklearn.metrics import f1_score

class CustomModel(nn.Module):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model
        self.clf = nn.Sequential(
            nn.SiLU(),
            nn.LazyLinear(25),
        )

    def forward(self, x, label=None):
        x = self.model(x)
        logits = self.clf(x)
        loss = None
        if label is not None:
            loss = nn.CrossEntropyLoss()(logits, label)
        probs = nn.LogSoftmax(dim=-1)(logits)
        return probs, loss