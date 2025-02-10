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
        # The einsum operation here maps one-hot encoded indices to their corresponding embedding vectors.
        # 
        # Let's break this down:
        # 1. The one-hot tensor (`one_hot`) has the shape [B, N, H, W], where N represents the number of embedding factors.
        # 2. The embedding weight (`self.embedding.weight`) is a matrix of size [N, D], where D is the embedding dimension.
        # 3. The einsum expression 'b n h w, n d -> b d h w' performs the following:
        #    - For each N in the one-hot tensor, it multiplies the corresponding embedding vector of size D.
        #    - As a result, each spatial position (H, W) in the input tensor is replaced with an embedding of size D.
        #
        # In simpler terms:                                                 
        # - This operation effectively replaces the one-hot encoded values in the input tensor with the corresponding 
        #   embedding vectors, producing a tensor of shape [B, D, H, W].
        # - The process transforms categorical data (one-hot encoded) into a continuous embedding space.

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
        # 이 코드는 우리가 익히 알고있는 vq-vae가 아니다. 
        # vq-vae는 실제 target y와 모델을 통해 예측 된 값 y'의 loss인 Reconstruction Loss,
        # encoding을 거친 잠재공간을 고정하고 Code-book을 학습하는 Embedding Loss,
        # Code-book을 고정하고, encoding을 거친 잠재공간을 학습하는 Commitment Loss로 이루어져있다. 
        # 하지만, 이 코드는 단순히 vae를 손댄 것으로, 
        # 기존 vae에서 목표로 하던 Gaussian으로의 근사를 uniform 으로 근사를 하는 것이다.
        # 그럼으로 이 코드에서는 kl_div을 통해 Regularization term을 구현하고 있다.
        # 잠재공간이 균등분포로 근사하게되면, 이산적인 이유:
        # 차원이 높은 데이터를 차원축소한다고 생각한다. 그렇다면 자연스럽게 어떤 부분에 있어서는 비슷한 녀석들끼리 클러스터링이 되고, 
        # 비슷한 모양을 지닌 녀석들끼리는 비슷하게 뭉칠 것이다. 
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