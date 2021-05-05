import numpy as np
import torch
import torch.nn as nn
from module.channel_wise_attention import channel_wise_attention
from module.CNN import CNN
from module.LSTM import LSTM
from module.self_attention import self_attention

class ACRNN(nn.Module):
    def __init__(self,input_channel_num,input_width,input_height,k,kh,kw,ks,ph,pw,ps,oc,hidden_dim,attention_size):
        super(ACRNN,self).__init__()
        self.H = input_channel_num
        self.W = input_width
        self.C = input_height
        self.reduce = k
        self.channel_wise_attention = channel_wise_attention(self.H,self.W,self.C,self.reduce)
        self.output_channel = oc
        self.kernel_height = kh
        self.kernel_width = kw
        self.kernel_stride = ks
        self.pooling_height = ph
        self.pooling_width = pw
        self.pooling_stride = ps
        self.cnn = CNN(self.H,self.C,self.W,self.kernel_height,self.kernel_width,self.kernel_stride,self.pooling_height,self.pooling_width,self.pooling_stride,self.output_channel)
        self.hidden_dim = hidden_dim
        self.lstm = LSTM(self.hidden_dim)
        self.hidden = attention_size
        self.self_attention = self_attention(self.hidden_dim,self.hidden)
        self.softmax = nn.Sequential(
            nn.Linear(n_hidden_state,num_labels),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        x_map, x_ca = self.channel_wise_attention(x)
        x_cn = self.cnn(x_ca)
        x_rn, x_c = self.lstm(x_cn)
        x_sa = self.self_attention(x_rn)
        x_sm = self.softmax(x_sa)
        return x_sm

# EEG sample
window_size = 384
n_channel = 32
# input
input_channel_num = 1
input_height = 32
input_width = 384
num_labels = 2

# channel wise attention
k = 15

# CNN
## conv
kernel_height = 32
kernel_width = 45
kernel_stride = 1
conv_channel_num = 40
## pooling
pooling_height = 1
pooling_width = 75
pooling_stride = 10

# LSTM
n_hidden_state = 64

# self attention
attention_size = 512

"""
a = torch.randn(800,1,384,32)
acrnn = ACRNN(input_channel_num,input_width,input_height,k,kernel_height,kernel_width,kernel_stride,pooling_height,pooling_width,pooling_stride,conv_channel_num,n_hidden_state,attention_size)
b = acrnn(a)
print(b)
"""