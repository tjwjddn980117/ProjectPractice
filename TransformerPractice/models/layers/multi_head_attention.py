from torch import nn
from models.layers.scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
    
    def split(self, tensor):
        # input param size: [batch_size, lenght, d_model]
        # output param size: [batch_size, n_head, lenght, d_tensor]
        batch_size, lenght, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        # reshape date with view and transpose
        tensor = tensor.view(batch_size, lenght, self.n_head, d_tensor).transpose(1,2)

        return tensor
    
    def concat(self, tensor):
        batch_size, head, lenght, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1,2).contiguous().view(batch_size,lenght,d_model)

        return tensor
    
    def forward(self, q, k, v, mask=None):
        # 1. get the q, k, v weight value
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split q, k, v
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. calculate attention score and attention value
        out, attention = self.attention(q,k,v,mask=mask)

        # 4. concat output and linear that
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. if you want to visualize attention map,
        # you should to visualize 'attention'

        return out