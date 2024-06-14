import torch
from torch import nn
import pytorch_lightning as L

from sklearn.metrics import f1_score

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
        x = self.model(x).pooler_output
        logits = self.clf(x)
        loss = None
        if label is not None:
            loss = nn.CrossEntropyLoss()(logits, label)
        probs = nn.LogSoftmax(dim=-1)(logits)
        return probs, loss

class LitCustomModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = CustomModel(model)
        self.validation_step_output = []

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return opt

    def training_step(self, batch, batch_idx=None):
        x = batch['pixel_values']
        label = batch['label']
        probs, loss = self.model(x, label)
        self.log(f"train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx=None):
        x = batch['pixel_values']
        label = batch['label']
        probs, loss = self.model(x, label)
        self.validation_step_output.append([probs, label, loss])  # Add loss to validation_step_output
        return loss  # Return loss value

    def predict_step(self, batch, batch_idx=None):
        x = batch['pixel_values']
        probs, _ = self.model(x)
        return probs

    def validation_epoch_end(self, step_output):
        pred = torch.cat([x for x, _, _ in self.validation_step_output]).cpu().detach().numpy().argmax(1)
        label = torch.cat([label for _, label, _ in self.validation_step_output]).cpu().detach().numpy()
        score = f1_score(label, pred, average='macro')

        # Calculate validation loss
        val_loss = torch.stack([loss for _, _, loss in self.validation_step_output]).mean()

        self.log("val_score", score)
        self.log("val_loss", val_loss)  # Log validation loss
        self.validation_step_output.clear()
        return score
