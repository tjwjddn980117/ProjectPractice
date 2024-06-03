import torch
from torch import nn
import torchvision
from torchinfo import summary

from utils.config import CFG

class SwinTransformerModel(nn.Module):
    def __init__(self, backbone_model, name='swin-transformer', 
                 num_classes=CFG.NUM_CLASSES, device=CFG.DEVICE):
        super(SwinTransformerModel, self).__init__()
        
        self.backbone_model = backbone_model
        self.device = device
        self.num_classes = num_classes
        self.name = name
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2, inplace=True), 
            nn.Linear(in_features=1000, out_features=256, bias=True),
            nn.GELU(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=num_classes, bias=False)
        ).to(device)
        
    def forward(self, image):
        vit_output = self.backbone_model(image)
        return self.classifier(vit_output)

def get_swin_b32_model(
    device: torch.device=CFG.NUM_CLASSES) -> nn.Module:
    # Set the manual seeds
    torch.manual_seed(CFG.SEED)
    torch.cuda.manual_seed(CFG.SEED)

    # Get model weights
    model_weights = (
        torchvision
        .models
        .Swin_V2_B_Weights
        .DEFAULT
    )
    
    # Get model and push to device
    model = (
        torchvision.models.swin_v2_b(
            weights=model_weights
        )
    ).to(device) 
    
    # Freeze Model Parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return model

# Get ViT model
vit_backbone = get_swin_b32_model(CFG.DEVICE)

vit_params = {
    'backbone_model'    : vit_backbone,
    'name'              : 'Swin-B32',
    'device'            : CFG.DEVICE
}

# Generate Model
vit_model = SwinTransformerModel(**vit_params)

# If using GPU T4 x2 setup, use this:
if CFG.NUM_DEVICES > 1:
    vit_model = nn.DataParallel(vit_model)

# View model summary
summary(
    model=vit_model, 
    input_size=(CFG.BATCH_SIZE, CFG.CHANNELS, CFG.WIDTH, CFG.HEIGHT),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"]
)

def Swin_B32():
    return vit_model

def print_Swin_B32():
    return summary