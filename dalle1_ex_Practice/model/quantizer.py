import torch
import torch.nn as nn
from einops import einsum, rearrange


class Quantizer(nn.Module):
    '''
    the class of Quantizer. 
    '''
    def __init__(self, num_embeddings, embedding_dim):
        '''
        the class of Qunatizer. 

        Arguments:
            num_embeddings (int): the number of embeddings (channels). 
            embedding_dim (int): the number of dimension. 

        Inputs:
            x (tensor): [B, C, H, W]. 
        
        Outputs:
            sampled (tensor): [B, D, H, W]. Newly sampled existing encoded x [B, C, H, wW through nn.Embedding's codebook. 
            kl_div (float): the difference between log_uniform and log_qy (KLD(P||Q)). 
            logits (tensor): [B, Pixel(H*W), C]. a rearrangement of the original existing x [B, C, H, W]. 
            log_qy (tensor): [B, Pixel(H*W), C]. log_softmax with logits. 
        '''
        super(Quantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(self.num_embeddings, embedding_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        # one_hot = [B, C, H, W]. 
        one_hot = torch.nn.functional.gumbel_softmax(x, tau=0.9, dim=1, hard=False) 
        # Changing the channel to the number of embedding_dim 
        #  and embedding it with N*N pictures as many as the number of embedding_dim 
        #  should be understood that this is the process of embedding itself to a picture. 
        # 
        # Let's say there are n (h*w) image channels. 
        #  If we proceed internally with these as (n*d) matrices,
        #  each of the n embedding factors is multiplied by n (h*w) according to the d embedding vector. 
        #  That is, n (h*w) images are multiplied by n embedding factors, which are d. This results in d (h*w) images. 
        sampled = einsum(one_hot, self.embedding.weight, 'b n h w, n d -> b d h w') 

        # Compute kl loss
        logits = rearrange(x, 'b n h w -> b (h w) n')
        # log_qy는 픽셀단위로 해당 픽셀이 어떤 channel을 가르킬 것인지에 대한 확률변수가 될 것이다. 
        # log_qy will be a random variable for which channel the pixel will teach in pixel units. 
        # log_qy : [B, pixel, channel]. 
        log_qy = torch.nn.functional.log_softmax(logits, dim=-1)
        # vq-vae는 인코딩된 잠재공간을 균등분포로 근사시킨다.
        log_uniform = torch.log(torch.tensor([1. / self.num_embeddings], device=torch.device(x.device)))
        # 잠재공간을 균등분포로 만들어 이진적인 성격을 지니게 한다.
        kl_div = torch.nn.functional.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target=True)
        # 이 코드에서 kl_div는 기존 vq-vae 논문에서의 commitment loss의 부분만 남아있는 모습니다. 
        # 이 코드에서는 기존 vq-vae 논문에서의 embedding loss는 없는 것을 알 수 있다. 
        return sampled, kl_div, logits, log_qy
    
    def quantize_indices(self, indices):
        '''
        The function for quantize. 

        Inputs:
            indices (tensor): [B, N, H, W]. 
        
        Returns:
            _ (tensor): [B, D, H, W]. 'D' is embedding_dim. 
        '''
        return einsum(indices, self.embedding.weight, 'b n h w, n d -> b d h w')