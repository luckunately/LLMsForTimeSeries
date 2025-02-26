import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from embed import DataEmbedding_wo_time_pos

# from  models.Attention import MultiHeadAttention
from torch.nn import MultiheadAttention
    
class Encoder_LLaTA(nn.Module):
    def __init__(self, input_dim , hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder_LLaTA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        x = self.linear(x)
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
        return x 

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
    
class PAttn(nn.Module):
    """
    pattn PAttn 
    
    Decomposition-Linear
    """
    def __init__(self, configs, device):
        super(PAttn, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_size = configs.patch_size 
        self.stride = configs.patch_size //2 
        
        self.d_model = configs.d_model
        self.method = configs.method
        # since classification, need to embed the data
        self.features = configs.features
        self.label_encoders = configs.label_encoders
        assert len(self.label_encoders) == self.features + 1
        self.embeddings = nn.ModuleList([torch.nn.Embedding(configs.seq_len, self.d_model) for le in (self.label_encoders)])
       
        self.basic_attn = MultiheadAttention(embed_dim =self.d_model, num_heads=8)
        output_embed_dim = len(self.label_encoders[-1].classes_)
        self.out_layer = nn.Linear(self.d_model * self.seq_len * self.features, configs.pred_len * output_embed_dim)
        
        # softmax for each pred_len
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        batch, features, seq_len = x.size()
        # embed the data
        x = torch.cat([self.embeddings[i](x[:, i, :]) for i in range(features)], dim=1)
        # now shape is [batch, features * history, d_model]

        x, _ = self.basic_attn( x ,x ,x )
        # flatten the data, but keep the batch dimension
        x = x.flatten(start_dim=1)
        
        # now shape is [batch, features * history, d_model] want [batch, pred_len, output_embed_dim]
        x = self.out_layer(x)
        
        x = x.reshape(batch, self.pred_len, -1)
        # do softmax for each pred_len
        x = self.softmax(x)

        return x  

# class PAttn_lstm(nn.Module):
#     """
#     pattn PAttn 
    
#     Decomposition-Linear
#     """
#     def __init__(self, configs, device):
#         super(PAttn_lstm, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.patch_size = configs.patch_size 
#         self.stride = configs.patch_size //2 
        
#         self.d_model = configs.d_model
#         self.method = configs.method
       
#         self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 2
#         self.padding_patch_layer = nn.ReplicationPad1d((0,  self.stride)) 
#         self.in_layer = nn.Linear(self.patch_size, self.d_model)
#         self.states = None
#         self.lstm1 = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=2, batch_first=True)
#         self.out_layer = nn.Linear(self.d_model * self.patch_num, configs.pred_len)
        
#         # print the size of each layer, follow the order of forward function
#         print('seq_len', self.seq_len)
#         print('patch_size', self.patch_size)
#         print('patch_num', self.patch_num)
#         print('d_model', self.d_model)
#         print('states', self.states)
#         print('lstm1', self.lstm1)
#         print('out_layer', self.out_layer)
#         print('method', self.method)
#         print('padding_patch_layer', self.padding_patch_layer)
#         print('in_layer', self.in_layer)
        
#         self.count = 0
        

#     def norm(self, x, dim =0, means= None , stdev=None):
#         if means is not None :  
#             return x * stdev + means
#         else : 
#             means = x.mean(dim, keepdim=True).detach()
#             x = x - means
#             stdev = torch.sqrt(torch.var(x, dim=dim, keepdim=True, unbiased=False)+ 1e-5).detach() 
#             x /= stdev
#             return x , means ,  stdev 
            
#     def forward(self, x):
#         if self.method == 'PAttn' and self.count != 0:
#             B , C = x.size(0) , x.size(1)
#             # [Batch, Channel, 336]
#             # x , means, stdev  = self.norm(x , dim=2)
#             # [Batch, Channel, 344]
#             x = self.padding_patch_layer(x)
#             # [Batch, Channel, 12, 16]
#             x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
#             # [Batch, Channel, 12, 768]
#             x = self.in_layer(x)
#             x =  rearrange(x, 'b c m -> (b c) m')
#             # if self.states is None:
#             #     self.states = (torch.zeros(2, B * C, self.d_model).to(x.device), torch.zeros(2, B * C, self.d_model).to(x.device))
#             x, _ = self.lstm1(x)
#             x =  rearrange(x, '(b c) m -> b (c m)' , b=B)
#             x = self.out_layer(x)
#             # x  = self.norm(x , means=means, stdev=stdev )
#             return x  
#         else:
#             self.count += 1
#             B , C = x.size(0) , x.size(1)
#             print(f"Input size: {x.size()}")
#             # [Batch, Channel, 336]
#             # x , means, stdev  = self.norm(x , dim=2)
#             # print(f"After normalization: {x.size()}")
#             # [Batch, Channel, 344]
#             x = self.padding_patch_layer(x)
#             print(f"After padding: {x.size()}")
#             # [Batch, Channel, 12, 16]
#             x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
#             print(f"After unfolding: {x.size()}")
#             # [Batch, Channel, 12, 768]
#             x = self.in_layer(x)
#             print(f"After in_layer: {x.size()}")
#             print(f"in_layer weights: {self.in_layer.weight.size()}")
#             x =  rearrange(x, 'b c m -> (b c) m')
#             # if self.states is None:
#             #     self.states = (torch.zeros(2, B * C, self.d_model).to(x.device), torch.zeros(2, B * C, self.d_model).to(x.device))
#             x, _ = self.lstm1(x)
#             print(f"After lstm1: {x.size()}")
#             # print(f"lstm1 weights: {self.lstm1.weight_ih_l0.size()}")
#             x =  rearrange(x, '(b c) m -> b (c m)' , b=B)
#             x = self.out_layer(x)
#             print(f"After out_layer: {x.size()}")
#             # print(f"out_layer weights: {self.out_layer.weight.size()}")
#             # x  = self.norm(x , means=means, stdev=stdev )
#             # print(f"After denormalization: {x.size()}")
#             return x  

            