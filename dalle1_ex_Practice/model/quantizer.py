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
            sampled (tensor): [B, D, H, W]. einsum with the dimension. 
            kl_div (float): the difference between log_uniform and log_qy (KLD(P||Q)). 
            logits (tensor): [B, Pixel(H*W), C]. 
            
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
        # log_uniform은 픽셀단위로 해상 픽셀어 어떤 channel을 가르킬 것인지 대해 확률은 모두 동일하다는 것을 보이는 타겟으로 균등분포이다.  
        # Log_uniform is a target that shows that the probabilities are all the same for which channel to teach the maritime pixel word in pixel units. 
        log_uniform = torch.log(torch.tensor([1. / self.num_embeddings], device=torch.device(x.device)))
        # log_uniform은 broadcasting되어 [B, pixel, channel]로 될 것이다. 
        # 이렇게 목적함수가 균등분포인 이유는, 결국 code book을 거쳐서 나온 embedding 들은 이진적인 성격을 가지고 있으며, 
        #  특정 image에 관해 특징을 encoding한 channel들을 code book을 거친 것이기에,
        #  결국 embedding을 한 후 나오는 결과 또한 동일한 code book index를 가져야 함에,
        #  픽셀이 sampling 되어져 나올 channel들은 모두 균등할 수 밖에 없다. 
        # The reason why the objective function is uniformly distributed is that after all, the embeddedings that come out through the code book have a binary character,
        # Channels that encode features for a specific image have gone through code books,
        # In the end, the result that comes out after embedding must also have the same code book index,
        # All the channels through which the pixels are sampled are bound to be equal. 
        kl_div = torch.nn.functional.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target=True)
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