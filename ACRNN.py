import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax
from module.channel_wise_attention import channel_wise_attention
from module.CNN import CNN
from module.LSTM import LSTM
from module.self_attention import self_attention


class ACRNN(nn.Module):
    def __init__(self,input_height):
        super(ACRNN,self).__init__()
        self.H = 1
        self.W = 384
        self.C = input_height
        self.reduce = 15
        self.channel_wise_attention = channel_wise_attention(self.H,self.W,self.C,self.reduce)
        self.output_channel = 40
        self.kernel_height = 32
        self.kernel_width = 45
        self.kernel_stride = 1
        self.pooling_height = 1
        self.pooling_width = 75
        self.pooling_stride = 10
        self.cnn = CNN(self.H,self.C,self.W,self.kernel_height,self.kernel_width,self.kernel_stride,self.pooling_height,self.pooling_width,self.pooling_stride,self.output_channel)
        self.hidden_dim = 64
        self.lstm = LSTM(self.hidden_dim)
        self.hidden = 512
        self.self_attention = self_attention(self.hidden_dim,self.hidden)
        self.num_labels = 2
        self.softmax = nn.Sequential(
            nn.Linear(self.hidden_dim,self.num_labels),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        x_map, x_ca = self.channel_wise_attention(x)
        x_cn = self.cnn(x_ca)
        x_rn, x_c = self.lstm(x_cn)
        y_map, x_sa = self.self_attention(x_rn)
        x_sm = self.softmax(x_sa)
        return x_map,y_map,x_sm

class A_CRNN(nn.Module):
    def __init__(self,input_height):
        super(A_CRNN,self).__init__()
        self.H = 1
        self.W = 384
        self.C = input_height
        self.reduce = 15
        self.channel_wise_attention = channel_wise_attention(self.H,self.W,self.C,self.reduce)
        self.output_channel = 40
        self.kernel_height = 32
        self.kernel_width = 45
        self.kernel_stride = 1
        self.pooling_height = 1
        self.pooling_width = 75
        self.pooling_stride = 10
        self.cnn = CNN(self.H,self.C,self.W,self.kernel_height,self.kernel_width,self.kernel_stride,self.pooling_height,self.pooling_width,self.pooling_stride,self.output_channel)
        self.hidden_dim = 64
        self.lstm = LSTM(self.hidden_dim)
        self.num_labels = 2
        self.softmax = nn.Sequential(
            nn.Linear(self.hidden_dim,self.num_labels),
            nn.Softmax(dim=2)
        )
    def forward(self,x):
        x_map, x_ca = self.channel_wise_attention(x)
        x_cn = self.cnn(x_ca)
        x_rn, x_c = self.lstm(x_cn)
        x_sm = self.softmax(x_rn)
        x_sm=x_sm.reshape(-1,2)
        return x_map,x_sm
"""
class A_CRNN(nn.Module):
    def __init__(self,input_height):
        super(A_CRNN,self).__init__()
        self.H = 1
        self.W = 384
        self.C = input_height
        self.reduce = 15
        self.channel_wise_attention = channel_wise_attention(self.H,self.W,self.C,self.reduce)
        self.output_channel = 40
        self.kernel_height = 32
        self.kernel_width = 45
        self.kernel_stride = 1
        self.pooling_height = 1
        self.pooling_width = 75
        self.pooling_stride = 10
        self.cnn = CNN(self.H,self.C,self.W,self.kernel_height,self.kernel_width,self.kernel_stride,self.pooling_height,self.pooling_width,self.pooling_stride,self.output_channel)
        self.hidden_dim = 64
        self.lstm = LSTM(self.hidden_dim)
    def forward(self,x):
        x_map, x_ca = self.channel_wise_attention(x)
        x_cn = self.cnn(x_ca)
        x_rn, x_c = self.lstm(x_cn)
        return x_rn

class Softmax(nn.Module):
    def __init__(self):
        super(Softmax,self).__init__()
        self.num_labels=2
        self.hidden_dim=64
        self.softmax = nn.Sequential(
            nn.Linear(self.hidden_dim,self.num_labels),
            nn.Softmax(dim=2)
        )
    def forward(self,x):
        y = self.softmax(x)
        y = y.reshape(-1,self.num_labels)
        return y
"""
class CRNN_A(nn.Module):
    def __init__(self,input_height):
        super(CRNN_A,self).__init__()
        self.H = 1
        self.W = 384
        self.C = input_height
        self.output_channel = 40
        self.kernel_height = 32
        self.kernel_width = 45
        self.kernel_stride = 1
        self.pooling_height = 1
        self.pooling_width = 75
        self.pooling_stride = 10
        self.cnn = CNN(self.H,self.C,self.W,self.kernel_height,self.kernel_width,self.kernel_stride,self.pooling_height,self.pooling_width,self.pooling_stride,self.output_channel)
        self.hidden_dim = 64
        self.lstm = LSTM(self.hidden_dim)
        self.hidden = 512
        self.self_attention = self_attention(self.hidden_dim,self.hidden)
        self.num_labels = 2
        self.softmax = nn.Sequential(
            nn.Linear(self.hidden_dim,self.num_labels),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        x_cn = self.cnn(x)
        x_rn, x_c = self.lstm(x_cn)
        y_map,x_sa = self.self_attention(x_rn)
        x_sm = self.softmax(x_sa)
        return x_sm