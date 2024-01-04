import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg

from utils import conv3x3, GLU

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
    def __init__(self, channel_num):
        '''
        This block work for ResBlock.

        Arguments:
            channel_num (int): the number of channel.

        Inputs:
            [B, channel_num, H, W]

        Outputs:
            [B, channel_num, H, W]
        '''
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
    def __init__(self):
        '''
        Conditioning Augumentation Network. 
        This is a key idea that enables backpropagation using the Reparametricization technique of VAEs.

        Inputs:
            text_embedding (nparray): [batch_size, cfg.TEXT.DIMENSION]

        Returns:
            c_code (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]. reparametrized numpy.
            mu (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]. just x[:, :self.ef_dim] after fc.
            logvar (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]. just x[:, self.ef_dim:] after fc.
        '''
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
            text_embedding (nparray): [batch_size, cfg.TEXT.DIMENSION].

        Outputs:
            mu (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]. just x[:, :self.ef_dim] after fc.
            logvar (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]. just x[:, self.ef_dim:] after fc.
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
            mu (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]. just x[:, :self.ef_dim] after fc.
            logvar (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]. just x[:, self.ef_dim:] after fc.

        Outputs:
            c_code (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]. reparametrized numpy.
        '''
        # Divide the log var by 2 
        #  and apply an exponential function to calculate the standard deviation (std).
        # std = [batch_size, cfg.GAN.EMBEDDING_DIM]
        std = logvar.mul(0.5).exp_()
        # eps = [batch_size, cfg.GAN.EMBEDDING_DIM]
        device = torch.device("cuda" if cfg.CUDA else "cpu") 
        eps = torch.randn_like(std).to(device)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar
    
class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        '''
        This is the code about initialize G_Stage. 
        We concatate z_code and c_code,
         then we decode the embedding containing sequence information as an image.
        
        Arguments:
            ngf (int): defined channel num.
         
        Inputs:
            z_code (nparray): [B, z_code]
            c_code (nparray): [B, c_code]
        
        Outputs:
            out_code (nparray): [B, ngf, 64, 64]
        '''
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if cfg.GAN.B_CONDITION:
            self.in_dim = cfg.GAN.Z_DIM + cfg.GAN.EMBEDDING_DIM
        else:
            self.in_dim = cfg.GAN.Z_DIM
        self.define_module()
    
    def define_module(self):
        '''
        this is the model of defining module.
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
            # in_code size is [B, (c_code + z_code)]
            in_code = torch.cat((c_code, z_code), 1)
        else:
            # in_code size is [B, z_code]
            in_code = z_code

        # [B, in_code]
        out_code = self.fc(in_code)
        # [B, [ngf x 4 x 4]]
        # However, we will assume that ngf is 16ngf for easy calculation.
        # so, the size will [B, [16ngf x 4 x 4]]
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # [B, 16ngf, 4, 4]
        out_code = self.upsample1(out_code)
        # [B, 8ngf, 8, 8]
        out_code = self.upsample2(out_code)
        # [B, 4ngf, 16, 16]
        out_code = self.upsample3(out_code)
        # [B, 2ngf, 32, 32]
        out_code = self.upsample4(out_code)
        # [B, ngf, 64, 64]

        return out_code
    
class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, num_residual=cfg.GAN.R_NUM):
        '''
        This is the model of next stage of Generation. 
        We residual in this layer, then we upsample just one time.

        Arguments:
            ngf (int): defined channel num.
            num_residual (int): defined iterate number of layers.

        Inputs:
            h_code (nparray): [B, img_dim, img_size, img_size]
            c_code (nparray): [B, emb_dim]

        Outputs:
            out_code (nparray): [B, (ngf/2), h_code_size*2, h_code_size*2]
        '''
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if cfg.GAN.B_CONDITION:
            self.ef_dim = cfg.GAN.EMBEDDING_DIM
        else:
            self.ef_dim = cfg.GAN.Z_DIM
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        '''
        This is the function for make layer.
        The layer don't change the size.
        '''
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        '''
        This is the model of defining module of layers.
        '''
        ngf = self.gf_dim
        efg = self.ef_dim
        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = upBlock(ngf, ngf // 2)
    
    def forward(self, h_code, c_code):
        # h_code will the data of image file 
        s_size = h_code.size(2)
        # c_code will resize the text to image
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        # we resize the c_code to img_size
        # for example, if ef_dim is [[1,2,3], [4,5,6]]
        #  then, it become [[[[1,1],[1,1]],[[2,2],[2,2]]...]
        c_code = c_code.repeat(1, 1, s_size, s_size)
        # state size (ngf+egf) x in_size x in_size
        # [B, (ngf+egf), in_size, in_size]
        h_c_code = torch.cat((c_code, h_code), 1)
        # state size ngf x in_size x in_size
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        # [B, (ngf/2), h_code_size*2, h_code_size*2]
        return out_code
    
class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        '''
        This is the function that Get Image from Generator.
        We adjust the input value of the weights of the convolution to the value between -1 and 1.

        Arguments:
            ngf (int): defined channel num.

        Inputs:
            h_code (nparray): [batch_size, in_planes, H, W]

        Outputs:
            out_img (nparray): [batch_size, out_planes, H, W]
        '''
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(self.gf_dim, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img

class G_NET(nn.Module):
    def __init__(self):
        '''
        This is the main Generation net. 
        The number of fake images stored varies depending on the depth.

        Input:
            z_code (nparray): [B, z_code]
            text_embedding (nparray): [batch_size, cfg.TEXT.DIMENSION]
        
        Outputs:
            fake_imgs (list[array]): the size of array is [batch_size, out_planes, H, W]
            mu (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]. just x[:, :self.ef_dim] after fc.
            logvar (nparray): [batch_size, cfg.GAN.EMBEDDING_DIM]. just x[:, self.ef_dim:] after fc.
        '''
        super(G_NET, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.define_module()

    def define_module(self):
        '''
        This is the function define the modules.
        '''
        if cfg.GAN.B_CONDITION:
            # conditioning argument net.
            self.ca_net = CA_NET()
        
        # If the branch get deeper, the layer get deeper too.
        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(self.gf_dim * 16)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(self.gf_dim)
            self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2)
            self.img_net3 = GET_IMAGE_G(self.gf_dim // 4)
        if cfg.TREE.BRANCH_NUM > 3: # Recommended structure (mainly limited by GPU memory), and not test yet
            self.h_net4 = NEXT_STAGE_G(self.gf_dim // 4, num_residual=1)
            self.img_net4 = GET_IMAGE_G(self.gf_dim // 8)
        if cfg.TREE.BRANCH_NUM > 4:
            self.h_net4 = NEXT_STAGE_G(self.gf_dim // 8, num_residual=1)
            self.img_net4 = GET_IMAGE_G(self.gf_dim // 16)

    def forward(self, z_code, text_embedding):
        if cfg.GAN.B_CONDITION and text_embedding is not None:
            c_code, mu, logvar = self.ca_net(text_embedding)
        else:
            c_code, mu, logvar = z_code, None, None

        fake_imgs = []
        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2 = self.h_net2(h_code1, c_code)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3 = self.h_net3(h_code2, c_code)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
        if cfg.TREE.BRANCH_NUM > 3:
            h_code4 = self.h_net4(h_code3, c_code)
            fake_img4 = self.img_net4(h_code4)
            fake_imgs.append(fake_img4)

        return fake_imgs, mu, logvar
