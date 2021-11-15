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
cn = CNN(1,32,384,32,45,1,1,75,10,40)
params = 0
for p in cn.parameters():
    if p.requires_grad:
        params += p.numel()
print(params)#57640
"""