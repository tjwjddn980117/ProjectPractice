from torchinfo import summary
import timm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pytorch_model import CustomModel
from utils.config import CFG

model = CustomModel(timm.create_model('eva_large_patch14_196.in22k_ft_in22k_in1k', pretrained=True))

print(summary(
        model=model, 
        input_size=(CFG.BATCH_SIZE, CFG.CHANNELS, CFG.WIDTH, CFG.HEIGHT),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
        ))

# # 동결시키고 싶은 블록의 인덱스 지정
# trainable_blocks = [0, 1, 2]
# 
# # 모든 파라미터 동결
# for param in model.parameters():
#     param.requires_grad = False
# 
# # 특정 블록만 학습 가능하도록 설정
# for idx in trainable_blocks:
#     for param in model.blocks[idx].parameters():
#         param.requires_grad = True
print()

# 모델의 모든 파라미터 이름과 학습 가능 여부 출력
for name, param in model.named_parameters():
    print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

target_parameters = ['model.blocks.17.', 'model.blocks.18', 'model.blocks.19', 'model.blocks.20', 'model.blocks.21',
                     'model.blocks.22', 'model.blocks.23', 'model.fc_norm', 'model.head', 'clf']

for name, param in model.named_parameters():
    if any(name.startswith(prefix) for prefix in target_parameters):
        print(name)
        param.requires_grad = True  # 동결
    else:
        param.requires_grad = False   # 학습 가능

print()
# 모델의 모든 파라미터 이름과 학습 가능 여부 재출력
for name, param in model.named_parameters():
    print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")