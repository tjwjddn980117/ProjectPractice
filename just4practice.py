import torch

# 예시로 y와 residual 텐서를 생성
y = torch.rand([32, 64, 64, 128])  # [B, H, W, C]
residual = torch.rand([32, 32, 32, 64])  # [B, H/2, W/2, C']

# 두 번째 차원을 따라 텐서를 연결 (axis=3)
result = torch.cat((y, residual), dim=3)
print(result.shape)
