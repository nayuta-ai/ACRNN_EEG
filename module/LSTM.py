import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
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
        h = hidden[1]
        c = cell[1]
        return h,c

model = LSTM(64)
a = torch.randn(800,1080)
b ,c= model(a)
print(b.shape)
print(c.shape)