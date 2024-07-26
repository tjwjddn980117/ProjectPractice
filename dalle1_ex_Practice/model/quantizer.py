import torch
import torch.nn as nn
from einops import einsum, rearrange


class Quantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Quantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(self.num_embeddings, embedding_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
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
        log_qy = torch.nn.functional.log_softmax(logits, dim=-1)
        log_uniform = torch.log(torch.tensor([1. / self.num_embeddings], device=torch.device(x.device)))
        kl_div = torch.nn.functional.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target=True)
        return sampled, kl_div, logits, log_qy
    
    def quantize_indices(self, indices):
        return einsum(indices, self.embedding.weight, 'b n h w, n d -> b d h w')