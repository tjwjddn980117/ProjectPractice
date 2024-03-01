import math
from torch import nn

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        '''
        Computing the Scale-Dot-Product with attention.

        Inputs:
            q(Tensor): [Batch_size, head, lenght, d_tensor].
            k(Tensor): [Batch_size, head, lenght, d_tensor].
            v(Tensor): [Batch_size, head, lenght, d_tensor].
            mask(Bool): Decoder should have a mask.
            e(float): not to be zero.

        '''
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input tensor size= [batch_size, head, lenght, d_tensor]
        batch_size, head, lenght, d_tensor = k.size()

        # 1. here is the dot production
        k_t = k.transpose(2,3)
        score = (q@k_t) / math.sqrt(d_tensor)

        # 2. masking
        if mask is not None:
            score = score.masked_fill_(mask == 0, -10000)

        # 3. softmax
        score = self.softmax(score)

        # 4. matmul
        v = score@v

        return v, score