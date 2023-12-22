import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
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

# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    '''
    This block work for up-sampling.
    
    Inputs:
        [B, in_planes, H, W]
    Outputs:
        [B, out_planes, H*2, W*2]
    '''
    block = nn.Sequential(
        # [B, in_planes, H, W]
        nn.Upsample(scale_factor=2, mode='nearest'),
        # [B, in_planes, H*2, W*2]
        conv3x3(in_planes, out_planes * 2),
        # [B, out_planes*2, H*2, W*2]
        nn.BatchNorm2d(out_planes * 2),
        # [B, out_planes*2, H*2, W*2]
        GLU()
        # [B, out_planes, H*2, W*2]
    )
    return block

# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    '''
    This block work for Relu.
    
    Inputs:
        [B, in_planes, H, W]
    Outputs:
        [B, out_planes, H, W]
    '''
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class ResBlock(nn.Module):
    '''
    This block work for ResBlock.
    
    Inputs:
        [B, channel_num, H, W]
    Outputs:
        [B, channel_num, H, W]
    '''
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )


    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    '''
    Conditioning Augumentation Network. 
    This is a key idea that enables backpropagation using the Reparametricization technique of VAEs.

    Inputs:
        [batch_size, cfg.TEXT.DIMENSION]

    Returns:
        [batch_size, cfg.GAN.EMBEDDING_DIM], 
        [batch_size, cfg.GAN.EMBEDDING_DIM],
        [batch_size, cfg.GAN.EMBEDDING_DIM].
    '''
    def __init__(self):
        super(CA_NET).__init__()
        self.t_dim = cfg.TEXT.DIMENSION
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        '''
        In encode part, we define mu and logvar randomly.
        We Initialize with 'parameters to be averaged' and 'parameters to be dispersed'
        and then update these parameters to values that naturally represent 
        mean and variance during the learning process.
        
        Inputs:
            [batch_size, cfg.TEXT.DIMENSION].

        Outputs:
            [batch_size, cfg.GAN.EMBEDDING_DIM], [batch_size, cfg.GAN.EMBEDDING_DIM].
        '''
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar
    
    def reparametrize(self, mu, logvar):
        '''
        The reparametrize method samples latent vectors using mean and variance. 
        This is a key idea that enables backpropagation using the Reparametricization technique of VAEs.
        Inputs:
            [batch_size, cfg.GAN.EMBEDDING_DIM], [batch_size, cfg.GAN.EMBEDDING_DIM].
        Outputs:
            [batch_size, cfg.GAN.EMBEDDING_DIM]
        '''
        # Divide the log var by 2 
        #  and apply an exponential function to calculate the standard deviation (std).
        # std = [batch_size, cfg.GAN.EMBEDDING_DIM]
        std = logvar.mul(0.5).exp_()
        # eps = [batch_size, cfg.GAN.EMBEDDING_DIM]
        if cfg.CUDA:
             eps = torch.randn_like(std).to('cuda')
        else:
             eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar
    
class INIT_STAGE_G(nn.Module):
    '''
    This is the code about initialize G_Stage
    '''
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if cfg.GAN.B_CONDITION:
            self.in_dim = cfg.GAN.Z_DIM + cfg.GAN.EMBEDDING_DIM
        else:
            self.in_dim = cfg.GAN.Z_DIM
        self.define_module()
    
    def define_module(self):
        '''
        this is the model of defining module
        '''
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU()
        )
        
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
    
    def forward(self, z_code, c_code=None):
        if cfg.GAN.B_CONDITION and c_code is not None:
            in_code = torch.cat((c_code, z_code), 1)
        else:
            in_code = z_code

        # state size 16ngf x 4 x 4
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size 8ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 4ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # state size 2ngf x 32 x 32
        out_code = self.upsample3(out_code)
        # state size ngf x 64 x 64
        out_code = self.upsample4(out_code)

        return out_code
    
class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, num_residual=cfg.GAN.R_NUM):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
