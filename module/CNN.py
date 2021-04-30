import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,ic,ih,iw,kh,kw,ks,ph,pw,ps,oc):
        super(CNN,self).__init__()
        # input
        self.input_channel = ic
        self.input_height =  ih
        self.input_width = iw
        self.output_channel = oc
        self.kernel_height = kh
        self.kernel_width = kw
        self.kernel_stride = ks
        self.pooling_height = ph
        self.pooling_width = pw
        self.pooling_stride = ps
        # CNN
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channel,self.output_channel,(self.kernel_height,self.kernel_width),self.kernel_stride),
            nn.ELU(),
            nn.MaxPool2d((self.pooling_height,self.pooling_width),self.pooling_stride)
        )
        # dropout
        self.dropout = nn.Dropout2d(p=0.5)
    def __call__(self,x):
        x = x.permute(0,1,3,2)
        c = self.conv(x)
        # c1 = c.reshape(800,-1)
        cd = self.dropout(c)
        return cd

"""
a = torch.randn(800,1,384,32)
# input 
input_channel_num = 1
input_height = 32
input_width = 384
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
cnn = CNN(input_channel_num,input_height,input_width,kernel_height,kernel_width,kernel_stride,pooling_height,pooling_width,pooling_stride,conv_channel_num)
b = cnn(a)
print(b.shape)
"""