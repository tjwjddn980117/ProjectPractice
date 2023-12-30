import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg

from utils import conv3x3

# ############## G networks ################################################
def Block3x3_leakRelu(in_planes, out_planes):
    '''
    This is the function that 3x3 leakyReLU. 
    the output has normalized.

    Inputs:
        in_planes (int): number of in_channel num.
        out_planes (int): number of out_channel num.

    Outputs:
        block (sequential): [B, in_planes, H, W] -> [B, out_planes, H, W].
    '''
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    '''
    This is the function of down sampling for feature extract.
    the output has normalized.

    Inputs:
        in_planes (int): number of in_channel num.
        out_planes (int): number of out_channel num.

    Outputs:
        block (sequential): [B, in_planes, H, W] -> [B, out_planes, H/2, W/2]
    '''
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    '''
    This is the function of encoding image.
    
    Inputs:
        ndf (int): defined channel num.
    
    Outputs:
        encode_img (Sequential): [B, 3, H, W] -> [B, ndf*8, H/16, W/16]
    '''
    encode_img = nn.Sequential(
        # [B, 3, H*16, W*16]
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # [B, ndf, H*8, W*8]
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # [B, ndf*2, H*4, W*4]
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # [B, ndf*4, H*2, W*2]
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
        # [B, ndf*8, H, W]
    )
    return encode_img

# for 64*64 image.
class D_NET64(nn.Module):
    def __init__(self):
        '''
        The class for discriminate 64 size. We discriminate after Sigmoid.
        If condition, we use h_c_code and x_code.
        But, it isn't condition, we only use x_code.

        Inputs:
            x_var (nparray):  [B, 3, 64, 64]. originial image.
            c_code (nparray): [B, emb_dim]. originial code.

        Outputs:
            output (list): 
                if condition : [2B]
                if uncondition : [B]
        '''
        super(D_NET64, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        # [B, ndf*8, 4, 4]
        self.img_code_s16 = encode_image_by_16times(ndf)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            # [B, 1, 1, 1]
            nn.Sigmoid())

        # if the conditioning block
        if cfg.GAN.B_CONDITION:
            # [B, ndf*8+efg, H, W]
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            # [B, ndf*8, H, W]
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                # [B, 1, 1, 1]
                nn.Sigmoid())
    
    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        # x_code should be [B, ndf*8, 4, 4]

        if cfg.GAN.B_CONDITION and c_code is not None:
            # we resize the c_code to img_size
            # for example, if ef_dim is [[1,2,3], [4,5,6]]
            #  then, it become [[[[1,1],[1,1]],[[2,2],[2,2]]...]
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # c_code should [B, egf, 4, 4]
            h_c_code = torch.cat((c_code, x_code), 1)
            # h_c_code should [B, ndf*8 + egf, 4, 4]
            h_c_code = self.jointConv(h_c_code)
            # h_c_code should [B, ndf*8, 4, 4]
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        # output should [B, 1, 1, 1]

        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]

# for 128*128 image.
class D_NET128(nn.Module):
    def __init__(self):
        '''
        The class for discriminate 128 size. We discriminate after Sigmoid.
        If condition, we use h_c_code and x_code.
        But, it isn't condition, we only use x_code.

        Inputs:
            x_var (nparray):  [B, 3, 128, 128]. originial image.
            c_code (nparray): [B, emb_dim]. originial code.

        Outputs:
            output (list): 
                if condition : [2B]
                if uncondition : [B]
        '''
        super(D_NET128, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        '''
        The function that define module.
        '''
        ndf = self.df_dim
        efg = self.ef_dim
        # [B, ndf*8, 8, 8]
        self.img_code_s16 = encode_image_by_16times(ndf)
        # [B, ndf*16, 4, 4]
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        # [B, ndf*8, 4, 4]
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            # [B, 1, 1, 1]
            nn.Sigmoid())

        # if the conditioning block
        if cfg.GAN.B_CONDITION:
            # [B, ndf*8+efg, H, W]
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            # [B, ndf*8, H, W]
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                # [B, 1, 1, 1]
                nn.Sigmoid())
            
    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)
        # x_code should be [B, ndf*8, 4, 4]

        if cfg.GAN.B_CONDITION and c_code is not None:
            # we resize the c_code to img_size
            # for example, if ef_dim is [[1,2,3], [4,5,6]]
            #  then, it become [[[[1,1],[1,1]],[[2,2],[2,2]]...]
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # c_code should [B, egf, 4, 4]
            h_c_code = torch.cat((c_code, x_code), 1)
            # h_c_code should [B, ndf*8 + egf, 4, 4]
            h_c_code = self.jointConv(h_c_code)
            # h_c_code should [B, ndf*8, 4, 4]
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        # output should [B, 1, 1, 1]

        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]

# for 256*256 image.
class D_NET256(nn.Module):
    def __init__(self):
        '''
        The class for discriminate 256 size. We discriminate after Sigmoid.
        If condition, we use h_c_code and x_code.
        But, it isn't condition, we only use x_code.

        Inputs:
            x_var (nparray):  [B, 3, 256, 256]. originial image.
            c_code (nparray): [B, emb_dim]. originial code.

        Outputs:
            output (list): 
                if condition : [2B]
                if uncondition : [B]
        '''
        super(D_NET256).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        '''
        The function that define module.
        '''
        ndf = self.df_dim
        efg = self.ef_dim
        # [B, ndf*8, 16, 16]
        self.img_code_s16 = encode_image_by_16times(ndf)
        # [B, ndf*16, 8, 8]
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        # [B, ndf*32, 4, 4]
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        # [B, ndf*16, 4, 4]
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        # [B, ndf*8, 4, 4]
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            # [B, 1, 1, 1]
            nn.Sigmoid())
        
        # if the conditioning block
        if cfg.GAN.B_CONDITION:
            # [B, ndf*8+efg, H, W]
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            # [B, ndf*8, H, W]
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                # [B, 1, 1, 1]
                nn.Sigmoid())
    
    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s64_1(x_code)
        x_code = self.img_code_s64_2(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            # we resize the c_code to img_size
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # c_code should [B, egf, 4, 4]
            h_c_code = torch.cat((c_code, x_code), 1)
            # h_c_code should [B, ndf*8 + egf, 4, 4]
            h_c_code = self.jointConv(h_c_code)
            # h_c_code should [B, ndf*8, 4, 4]
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        # output should [B, 1, 1, 1]

        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]

# for 512*512 image.
class D_NET512(nn.Module):
    def __init__(self):
        '''
        The class for discriminate 512 size. We discriminate after Sigmoid.
        If condition, we use h_c_code and x_code.
        But, it isn't condition, we only use x_code.

        Inputs:
            x_var (nparray):  [B, 3, 512, 512]. originial image.
            c_code (nparray): [B, emb_dim]. originial code.

        Outputs:
            output (list): 
                if condition : [2B]
                if uncondition : [B]
        '''
        super(D_NET512).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        '''
        The function that define module.
        '''
        ndf = self.df_dim
        efg = self.ef_dim

        # [B, ndf*8, 32, 32]
        self.img_code_s16 = encode_image_by_16times(ndf)
        # [B, ndf*16, 16, 16]
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        # [B, ndf*32, 8, 8]
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        # [B, ndf*64, 4, 4]
        self.img_code_s128 = downBlock(ndf * 32, ndf * 64)
        # [B, ndf*32, 4, 4]
        self.img_code_s128_1 = Block3x3_leakRelu(ndf * 64, ndf * 32)
        # [B, ndf*16, 4, 4]
        self.img_code_s128_2 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        # [B, ndf*8, 4, 4]
        self.img_code_s128_3 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            # [B, 1, 1, 1]
            nn.Sigmoid())
        
        # if the conditioning block
        if cfg.GAN.B_CONDITION:
            # [B, ndf*8+efg, H, W]
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            # [B, ndf*8, H, W]
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                # [B, 1, 1, 1]
                nn.Sigmoid())
        
    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s128(x_code)
        x_code = self.img_code_s128_1(x_code)
        x_code = self.img_code_s128_2(x_code)
        x_code = self.img_code_s128_3(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            # we resize the c_code to img_size
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # c_code should [B, egf, 4, 4]
            h_c_code = torch.cat((c_code, x_code), 1)
            # h_c_code should [B, ndf*8 + egf, 4, 4]
            h_c_code = self.jointConv(h_c_code)
            # h_c_code should [B, ndf*8, 4, 4]
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        # output should [B, 1, 1, 1]

        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]

# for 1024*1024 image.
class D_NET1024(nn.Module):
    def __init__(self):
        '''
        The class for discriminate 1024 size. We discriminate after Sigmoid.
        If condition, we use h_c_code and x_code.
        But, it isn't condition, we only use x_code.

        Inputs:
            x_var (nparray):  [B, 3, 1024, 1024]. originial image.
            c_code (nparray): [B, emb_dim]. originial code.

        Outputs:
            output (list): 
                if condition : [2B]
                if uncondition : [B]
        '''
        super(D_NET1024).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        '''
        The function that define module.
        '''
        ndf = self.df_dim
        efg = self.ef_dim
        # [B, ndf*8, 64, 64]
        self.img_code_s16 = encode_image_by_16times(ndf)
        # [B, ndf*16, 32, 32]
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        # [B, ndf*32, 16, 16]
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        # [B, ndf*64, 8, 8]
        self.img_code_s128 = downBlock(ndf * 32, ndf * 64)
        # [B, ndf*128, 4, 4]
        self.img_code_s256 = downBlock(ndf * 64, ndf * 128)
        # [B, ndf*64, 4, 4]
        self.img_code_s256_1 = Block3x3_leakRelu(ndf * 128, ndf * 64)
        # [B, ndf*32, 4, 4]
        self.img_code_s256_2 = Block3x3_leakRelu(ndf * 64, ndf * 32)
        # [B, ndf*16, 4, 4]
        self.img_code_s256_3 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        # [B, ndf*8, 4, 4]
        self.img_code_s256_4 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            # [B, 1, 1, 1]
            nn.Sigmoid())
        
        # if the conditioning block
        if cfg.GAN.B_CONDITION:
            # [B, ndf*8+efg, H, W]
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            # [B, ndf*8, H, W]
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                # [B, 1, 1, 1]
                nn.Sigmoid())
            
    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s128(x_code)
        x_code = self.img_code_s256(x_code)
        x_code = self.img_code_s256_1(x_code)
        x_code = self.img_code_s256_2(x_code)
        x_code = self.img_code_s256_3(x_code)
        x_code = self.img_code_s256_4(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            # we resize the c_code to img_size
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # c_code should [B, egf, 4, 4]
            h_c_code = torch.cat((c_code, x_code), 1)
            # h_c_code should [B, ndf*8 + egf, 4, 4]
            h_c_code = self.jointConv(h_c_code)
            # h_c_code should [B, ndf*8, 4, 4]
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        # output should [B, 1, 1, 1]

        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]