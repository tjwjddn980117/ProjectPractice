import torch

from torch import nn
from torch.nn import functional as F
model_args = {
    "in_channels": 3,
    "num_hiddens": 128,
    "num_downsampling_layers": 2,
    "num_residual_layers": 2,
    "num_residual_hiddens": 32,
    "embedding_dim": 64,
    "num_embeddings": 512,
    "decay": 0.99,
    "epsilon": 1e-5,
}

class SonnetExponentialMovingAverage(nn.Module):
    # See: https://github.com/deepmind/sonnet/blob/5cbfdc356962d9b6198d5b63f0826a80acfdf35b/sonnet/src/moving_averages.py#L25.
    # They do *not* use the exponential moving average updates described in Appendix A.1
    # of "Neural Discrete Representation Learning". 
    def __init__(self, decay, shape):
        '''
        This code block implements a variant of the Exponential Moving Average (EMA), 
        which computes the moving average of weights during the learning process to adjust the weights in the model to be unbiased.

        Arguments: 
            decay(float): Weight for the exponential moving average. 
                Higher decay rates retain more historical information, and lower decay rates give greater weight to recent values. 
            shape(Tensor): shape of input. 
        
        Inputs:
            values(Tensor): [shape]. 
        '''
        super(SonnetExponentialMovingAverage, self).__init__()
        self.decay = decay
        self.counter = 0
        # unpacking
        self.register_buffer("hidden", torch.zeros(*shape))
        self.register_buffer("average", torch.zeros(*shape))

    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay ** self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average

class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, use_ema, decay, epsilon):
        '''
        This is VectorQuantizer. 

        Arguments:
            embedding_dim(int): dimention of each embedding. 
            num_embeddings(int): number of embeddings. you can think it to the number of words in dictionary. 
            use_ema (bool): the bool type for checking use ema. 
            decay ( ): the parammeter using with SonnetEMA. 
            epsilon ( ): the parammeter using in equation. 
        
        Inputs:
            x(Tensor): [B, embedding_dim, H', W'].

        Outputs:
            quantized_x (Tensor): the tensor with qunatized. 
            dictionary_loss (item.float): the loss about dictionary. 
            commitment_loss (item.float): the loss about commitment. 
            encoding_indices.view(x.shape[0], -1) (Tensor): the indices

        '''
        super(VectorQuantizer,self).__init__()
        # See Section 3 of "Neural Discrete Representation Learning" and:
        # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L142.

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.use_ema = use_ema
        
        # Weight for the exponential moving average.
        self.decay = decay
        # Small constant to avoid numerical instability in embedding updates.
        self.epsilon = epsilon

        # Dictionary embeddings.
        # it is for init the embedding.
        limit = 3 ** 0.5
        e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(-limit, limit) # (-limit, limit)
        if use_ema: 
            # it dosen't update with back-propagation.
            # because, embedding dictionary will update until EMA(Exponential Moving Average).
            self.register_buffer("e_i_ts", e_i_ts)
        else: 
            # it do update with back-propagation.
            self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))

        # Exponential moving average of the cluster counts.
        self.N_i_ts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        # Exponential moving average of the embeddings.
        self.m_i_ts = SonnetExponentialMovingAverage(decay, e_i_ts.shape)

    def forward(self, x):
        # x => [B, embedding_dim, H', W']
        # flat_x => [B*H'*W', embedding_dim]
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        distances = (
            (flat_x ** 2).sum(1, keepdim=True) # [B*H'*W', 1]
            - 2 * flat_x @ self.e_i_ts # [B*H'*W', embedding_dim] @ [embedding_dim, num_embedding] = [B*H'*W', num_embedding]
            + (self.e_i_ts ** 2).sum(0, keepdim=True) # [1, num_embedding]
        ) # >> [B*H'*W', num_embedding]

        # encoding_indices >> the index of minimum distance
        # encoding_indices >> [B*H'*W', ]
        encoding_indices = distances.argmin(1)
        quantized_x = F.embedding(
            encoding_indices.view(x.shape[0], *x.shape[2:]), self.e_i_ts.transpose(0, 1)
        ).permute(0, 3, 1, 2) # >> [B, embedding_dim, H', W']

        # See second term of Equation (3). 
        if not self.use_ema:
            dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()
            # dictionary_loss.item() >> float
        else:
            dictionary_loss = None

        # See third term of Equation (3). 
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()
        # commitment_loss.item() >> float

        # Straight-through gradient. See Section 3.2. 
        # I don't know what's the reason of this. 
        quantized_x = x + (quantized_x - x).detach()

        if self.use_ema and self.training:
            with torch.no_grad():
                # See Appendix A.1 of "Neural Discrete Representation Learning".

                # Cluster counts.
                encoding_one_hots = F.one_hot(
                    encoding_indices, self.num_embeddings
                ).type(flat_x.dtype) # encoding_one_hots >> [B*H'*W', num_embeddings] 
                n_i_ts = encoding_one_hots.sum(0) # n_i_ts = [num_embeddings] 
                # Updated exponential moving average of the cluster counts. 
                # See Equation (6).
                self.N_i_ts(n_i_ts) # return also [num_embeddings] 

                # Exponential moving average of the embeddings. See Equation (7). 
                # embed_sums = [embedding_dim, num_embeddings] 
                embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                self.m_i_ts(embed_sums) # return also [embedding_dim, num_embeddings]

                # This is kind of weird. 
                # Compare: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L270
                # and Equation (8). 
                N_i_ts_sum = self.N_i_ts.average.sum()
                N_i_ts_stable = (
                    (self.N_i_ts.average + self.epsilon)
                    / (N_i_ts_sum + self.num_embeddings * self.epsilon)
                    * N_i_ts_sum
                )
                self.e_i_ts = self.m_i_ts.average / N_i_ts_stable.unsqueeze(0)

        return (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            encoding_indices.view(x.shape[0], -1),
        )