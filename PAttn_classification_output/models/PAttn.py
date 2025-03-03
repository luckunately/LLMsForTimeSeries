import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from embed import DataEmbedding_wo_time_pos

# from  models.Attention import MultiHeadAttention
from torch.nn import MultiheadAttention, LSTM
    
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
    
    
# class PAttn(nn.Module):
#     """
#     pattn PAttn 
    
#     Decomposition-Linear
#     """
#     def __init__(self, configs, device):
#         super(PAttn, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.patch_size = configs.patch_size 
#         self.stride = configs.patch_size //2 
        
#         self.d_model = configs.d_model
#         self.method = configs.method
#         # since classification, need to embed the data
#         self.features = configs.features
#         self.label_encoders = configs.label_encoders
#         assert len(self.label_encoders) == self.features + 1
#         # self.embeddings = nn.ModuleList([torch.nn.Embedding(configs.seq_len, self.d_model) for le in (self.label_encoders)])
#         self.in_layer = nn.Linear(self.features * self.seq_len, self.d_model)
       
#         # self.basic_attn = MultiheadAttention(embed_dim =self.d_model, num_heads=8)
#         self.lstm = LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=2, batch_first=True)
        
        
#         output_embed_dim = len(self.label_encoders[-1].classes_)
#         self.out_layer = nn.Linear(self.d_model, configs.pred_len * output_embed_dim)
        
#         # softmax for each pred_len
#         self.softmax = nn.Softmax(dim=2)
        
#     def forward(self, x):
#         batch, features, seq_len = x.size()
#         # embed the data
#         # x = torch.cat([self.embeddings[i](x[:, i, :]) for i in range(features)], dim=1)
#         x = x.flatten(start_dim=1).float()
#         x = self.in_layer(x)
#         # now shape is [batch, features * history, d_model]

#         # x, _ = self.basic_attn( x ,x ,x )
#         x, _ = self.lstm(x)
#         # flatten the data, but keep the batch dimension
#         x = x.flatten(start_dim=1)
        
#         # now shape is [batch, features * history, d_model] want [batch, pred_len, output_embed_dim]
#         x = self.out_layer(x)
        
#         x = x.reshape(batch, self.pred_len, -1)
#         # do softmax for each pred_len
#         x = self.softmax(x)

#         return x  

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
        self.padding_patch_layer = nn.ReplicationPad1d((0,  self.stride)) 
        
        assert len(self.label_encoders) == self.features + 1
        # self.embeddings = nn.ModuleList([torch.nn.Embedding(configs.seq_len, self.d_model) for le in (self.label_encoders)])
        self.in_layer = nn.Linear(self.features * self.seq_len, self.d_model)
       
        # self.basic_attn = MultiheadAttention(embed_dim =self.d_model, num_heads=8)
        self.lstm = LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=2, batch_first=True)
        
        
        output_embed_dim = len(self.label_encoders[-1].classes_)
        self.out_layer = nn.Linear(self.d_model * self.features, configs.pred_len * output_embed_dim)
        
        # softmax for each pred_len
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        batch, features, seq_len = x.size()
        # embed the data
        x , means, stdev  = self.norm(x , dim=2)
        x = self.padding_patch_layer(x)
        # [Batch, Channel, 12, 16]
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # [Batch, Channel, 4, 16]-> [Batch, Channel, 64]
        x = rearrange(x, 'b c m l -> b c (m l)')
        
        # x = self.in_layer(x)
        x = self.in_layer(x.to(self.in_layer.weight.dtype))
        
        # now shape is [batch, features * history, d_model]

        # x, _ = self.basic_attn( x ,x ,x )
        x, _ = self.lstm(x)
        # flatten the data, but keep the batch dimension
        x = x.flatten(start_dim=1)
        
        # now shape is [batch, features * history, d_model] want [batch, pred_len, output_embed_dim]
        x = self.out_layer(x)
        
        x = x.reshape(batch, self.pred_len, -1)
        # do softmax for each pred_len
        x = self.softmax(x)

        return x  
    
    def norm(self, x, dim =0, means= None , stdev=None):
        if means is not None :  
            return x * stdev + means
        else : 
            means = x.mean(dim, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=dim, keepdim=True, unbiased=False)+ 1e-5).detach() 
            x /= stdev
            return x , means ,  stdev 

# class PAttn(nn.Module):
#     """
#     pattn PAttn 
    
#     Decomposition-Linear
#     """
#     def __init__(self, configs, device):
#         super(PAttn, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.patch_size = configs.patch_size 
#         self.stride = configs.patch_size //2 
        
#         self.d_model = configs.d_model
#         self.method = configs.method
       
#         self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 2
#         self.padding_patch_layer = nn.ReplicationPad1d((0,  self.stride)) 
#         self.in_layer = nn.Linear(self.patch_size, self.d_model)
#         self.basic_attn = MultiheadAttention(self.d_model, num_heads=8)
#         self.out_layer = nn.Linear(self.d_model * self.patch_num, configs.pred_len)
        

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
#         if self.method == 'PAttn':
#             B , C = x.size(0) , x.size(1)
#             # [Batch, Channel, 336]
#             x , means, stdev  = self.norm(x , dim=2)
#             # [Batch, Channel, 344]
#             x = self.padding_patch_layer(x)
#             # [Batch, Channel, 12, 16]
#             x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
#             # [Batch, Channel, 12, 768]
#             x = self.in_layer(x.to(self.in_layer.weight.dtype))
#             x =  rearrange(x, 'b c m l -> (b c) m l')
#             x , _ = self.basic_attn( x ,x ,x )
#             x =  rearrange(x, '(b c) m l -> b c (m l)' , b=B , c=C)
#             x = self.out_layer(x)
#             x  = self.norm(x , means=means, stdev=stdev )
#             return x  
            