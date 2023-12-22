import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo

# ############################## For Compute inception score ##############################
# Besides the inception score computed by pretrained model, especially for fine-grained datasets (such as birds, bedroom),
#  it is also good to compute inception score using fine-tuned model and manually examine the image quality.

class INCEPTION_V3(nn.Module):
    '''
    INCEPTION_V3 is the pre-trained model.
    We don't update this model in this code.
    This model has a structure of normalization, upsampling, and sigmoid.

    Inputs:
        [batch, 3, 299, 299]

    Returns:
        [batch, 1000]
    '''
    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        # print(next(model.parameters()).data)
        state_dict = \
            model_zoo.load_url(url, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        
        # don't canculate gradient desenct.
        for param in self.model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(next(self.model.parameters()).data)
        # print(self.model)

    def forward(self, input):
        # [-1.0, 1.0] --> [0, 1.0]
        # this can resize the range [-1.0, 1.0] to [0, 1.0]
        x = input * 0.5 + 0.5
        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] for each channel
        # --> make mean = 0, std = 1 for normalize
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        #
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.model(x)
        # Inception v3's output should be [batch, 1000].
        # it is more likely classification model.
        x = nn.Softmax()(x)
        return x

class GLU(nn.Module):
    '''
    GLU is the activate function named 'Gated Linear Unit'.
    GLUs can control the flow of information about some of the inputs.
    This helps the neural network selectively focus important information.
    This helps the model learn more complex patterns 
        and filter out unnecessary information.
    
    Inputs: 
        [batch, channels]
    
    Outputs:
        [batch, channels/2]
    '''
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1) # check channel is add
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])
    
def conv3x3(in_planes, out_planes):
    '''
    3x3 convolution with padding
    
    Inputs:
        [batch_size, in_planes, H, W]
    Outputs:
        [batch_size, out_planes, H, W]
    '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)