import torch
import torch.nn as nn

# 입력 데이터 예시 (배치 크기, 채널 수, 높이, 너비)
input_data = torch.rand(1, 1, 4, 4)  # 1개의 데이터, 1채널, 4x4 크기의 이미지

# MaxPooling2d 레이어 생성 (stride를 명시하지 않음)
max_pooling_layer = nn.MaxPool2d(kernel_size=2)

# 입력 데이터에 MaxPooling2d 적용
output_data = max_pooling_layer(input_data)

# 결과 출력
print("Input shape:", input_data.shape)
print("Output shape:", output_data.shape)