import torch
import torchfile
#import matplotlib.pyplot as plt

# .t7 파일 로드
path_imgs ='001.Black_footed_Albatross.t7'

data = torch.Tensor(torchfile.load(path_imgs))
# 첫 번째 이미지 로드
#image = data[0]

print(type(data))
print("Shape of the NumPy array:", data.shape)
print(data)

# 이미지 출력
#plt.show()