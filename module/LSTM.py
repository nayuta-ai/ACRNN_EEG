import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self,hidden_dim):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        # lstm
        self.lstm = nn.LSTM(
            input_size=1080,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True
            )

    def forward(self,x,hidden0=None):
        x = x.reshape(-1,1,1080)
        q ,(hidden,cell) = self.lstm(x)
        h = hidden[1].reshape(-1,1,64)
        c = cell[1].reshape(-1,1,64)
        return h,c
"""
ls = LSTM(64)
params = 0
for p in ls.parameters():
    if p.requires_grad:
        params += p.numel()
print(params)#326656
"""