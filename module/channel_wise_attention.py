import torch.nn as nn
import torch

class channel_wise_attention(nn.Module):
    def __init__(self,H,W,C,reduce):
        super(channel_wise_attention,self).__init__()
        self.H = H
        self.W = W
        self.C = C
        self.r = reduce
        # fc layer
        self.fc = nn.Sequential(
            nn.Linear(self.C,self.r),
            nn.Tanh(),
            nn.Linear(self.r,self.C)
        )
        # softmax
        self.softmax = nn.Softmax(dim=3)

    def forward(self,x):
        # mean pooling
        x1 = x.permute(0,3,1,2)
        mean = nn.AvgPool2d((1,384))
        feature_map = mean(x1).permute(0,2,3,1)
        # FC Layer
        # feature_map : [800,1,1,C]
        feature_map_fc = self.fc(feature_map)
        
        # softmax
        v = self.softmax(feature_map_fc)
        # channel_wise_attention
        v = v.reshape(-1,self.C)
        vr = torch.reshape(torch.cat([v]*(self.H*self.W),axis=1),[-1,self.H,self.W,self.C])
        channel_wise_attention_fm = x * vr
        return v, channel_wise_attention_fm
"""
ca = channel_wise_attention(1,384,32,15)
params = 0
for p in ca.parameters():
    if p.requires_grad:
        params += p.numel()
print(params)#1007
"""