import torch
import torch.nn as nn
import torch.nn.functional as F

import math

max_seq_lenght = 100
total_word_num = 100

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_num=512, head_num=8):
        super().__init__()
        self.head_num = head_num
        self.dim_num = dim_num

        self.qurey_embed = nn.Linear(dim_num, dim_num)
        self.key_embed = nn.Linear(dim_num,dim_num)
        self.value_embed = nn.Linear(dim_num,dim_num)
        self.output_embed = nn.Linear(dim_num,dim_num)

    def scaled_dot_prodect_attention(self, q, k, v, mask=None):
        d_k = k.size()[-1] # hidden size
        k_transpose = torch.transpose(k, 3, 2)
        output = torch.matmul(q, k_transpose)
        output = output / math.sqrt(d_k)

        if mask is not None: # if you had masking
            output = output.masked_fill(mask.unsqueeze(1).unsqueeze(-1),0)


        output = F.softmax(output, -1)
        output = torch.matmul(output,v)

        return output
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size()[0]

        q = self.qurey_embed(q).view(batch_size, -1, self.head_num, self.dim_num // self.head_num).transpose(1, 2)
        k = self.key_embed(k).view(batch_size -1, self.head_num, self.dim_num // self.head_num).transpose(1, 2)
        v = self.value_embed(v).view(batch_size, -1, self.head_num, self.dim_num // self.head_num).transpose(1, 2)

        output = self.scaled_dot_prodect_attention(q,k,v,mask)
        batch_num, head_num, seq_num, hidden_num = output.size()
        output = torch.transpose(output, 1, 2).contiguous().view((batch_size), -1, hidden_num * self.head_num)

        return output
    
class FeedForward(nn.Module):
    def __init__(self, dim_num=256):
        super().__init__()
        self.layer1 = nn.Linear(dim_num, dim_num * 4)
        self.layer2 = nn.Linear(dim_num * 4, dim_num)
    
    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(F.relu(output))
    
        return output
    
class AddNormLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def layer_norm(self, input):
        mean = torch.mean(input, dim=-1, keepdim=True)
        std = torch.std(input, dim=-1, keepdim=True)
        output = (input - mean)/std
        return output
    
    def forward(self, input, residual):
        return residual + self.layer_norm(input)

class Encoder(nn.Module):
    def __init__(self, dim_num=512):
        super().__init__()
        self.multihead = MultiHeadAttention(dim_num=dim_num)
        self.residual_layer1 = AddNormLayer()
        self.feed_forward = FeedForward(dim_num=dim_num)
        self.residual_layer2 = AddNormLayer()

    def forward(self, q, k, v):
        multihead_output = self.multihead(v,q,k)
        residual_layer1 = self.residual_layer1(multihead_output, q)
        feedforward_output = self.feed_forward(residual_layer1)
        output = self.residual_layer2(feedforward_output, residual_layer1)

        return output
    
class Decoder(nn.Module):
    def __init__(self, dim_num=512):
        super().__init__()
        self.masked_multihead = MultiHeadAttention(dim_num=dim_num)
        self.residual_layer1 = AddNormLayer()
        self.multihead = MultiHeadAttention(dim_num=dim_num)
        self.residual_layer2 = AddNormLayer()
        self.feedforward = FeedForward(dim_num=dim_num)
        self.residual_layer3 = AddNormLayer()

    def forward(self, o_q, o_k, o_v, encoder_output, mask):
        masked_multihead_output = self.masked_multihead(o_v, o_q, o_k, mask)
        residual_layer1_output = self.residual_layer1(masked_multihead_output,o_q)
        multihead_output = self.multihead(encoder_output, encoder_output, residual_layer1_output, mask)
        residual_layer2_output = self.residual_layer2(multihead_output, residual_layer1_output)
        feedforward_output = self.feedforward(residual_layer2_output)
        output = self.residual_layer3(feedforward_output, residual_layer2_output)
        
        return output

class Transformer(nn.Module):
    def __init__(self, encoder_num = 6, decoder_num=6, hidden_dim=512,\
                  max_encoder_seq_lenght=100, max_decoder_seq_lenght=100):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.max_encoder_seq_lenght=100
        self.max_decoder_seq_lenght=100

        self.input_data_embed = nn.Embedding(total_word_num, self.hidden_dim)
        self.Encoders = [Encoder(dim_num=hidden_dim) for _ in range(encoder_num)]

        self.output_data_embed = nn.Embedding(total_word_num, self.hidden_dim)
        self.Decoders = [Decoder(dim_num=hidden_dim) for _ in range(decoder_num)]

        self.last_linear_layer = nn.Linear(self.hidden_dim, max_seq_lenght)

    
    def position_encoding(self, position_max_lenght=100):
        position = torch.arange(0,position_max_lenght, dtype=torch.float).unsqueeze(1)
        pe = torch.zeros(position_max_lenght, self.hidden_dim)
        div_term = torch.pow(torch.ones(self.hidden_dim) // 2).fill_(10000),\
                                torch.arange(0, self.hidden_dim ,2 ) / torch.tensor(self.hidden_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        return pe

    def forward(self, input, output, mask):
        input_embed = self.input_data_embed(input)
        input_embed += self.position_encoding(self.max_encoder_seq_lenght)
        q, k, v = input_embed, input_embed, input_embed

        for encoder in self.Encoders:
            encoder_output = encoder(q,k,v)
            q = encoder_output
            k = encoder_output
            v = encoder_output
        
        output_embed = self.output_data_embed(output)
        output_embed += self.position_encoding(self.max_decoder_seq_lenght)
        output_embed = output_embed.masked_fill(mask.unsqueeze(-1), 0)
        d_q, d_k, d_v = output_embed, output_embed, output_embed

        for decoder in self.Decoders:
            decoder_output = decoder(d_q,d_k,d_v,encoder_output, mask)
            d_q = decoder_output
            d_k = decoder_output
            d_v = decoder_output
        
        output = F.softmax(self.last_linear_layer(decoder_output), dim=-1)
        return output

if __name__=='__main__':
    model = Transformer()
    input = torch.randint(low=0, high=max_seq_lenght, size=(64, max_seq_lenght), dtype=torch.long)
    output = torch.randint(low=0, hight=max_seq_lenght, size=(64, max_seq_lenght), dtype=torch.long)
    mask = torch.zeros((64, max_seq_lenght))
    mask[:,:30] = 1

    output= model(input,output,mask)
    _, pred = torch.max(output, dim=-1)
    print(pred.shape)

