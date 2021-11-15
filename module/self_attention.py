import torch
import torch.nn as nn
import numpy as np

class dense(nn.Module):
    def __init__(self,input_dim1,input_dim2,hidden_dim,
    activation=lambda x:x):
        super().__init__()
        self.W1 = nn.Parameter(torch.Tensor(np.random.normal(size=(input_dim1,hidden_dim))))
        self.W2 = nn.Parameter(torch.Tensor(np.random.normal(size=(input_dim2,hidden_dim))))
        self.b = nn.Parameter(torch.Tensor(np.zeros(hidden_dim)))
        self.activation = activation
        self.vector = nn.Linear(input_dim2,input_dim2)
    def forward(self,x):
        y = self.vector(x)
        return self.activation(torch.matmul(x,self.W1)+torch.matmul(y,self.W2)+self.b)

class self_attention(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(self_attention,self).__init__()
        self.q = input_dim
        self.k = input_dim
        self.hidden = hidden_dim
        self.dense = dense(self.q,self.k,self.k)
        self.self_attention = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.k,self.k)
        )
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout()
    
    def forward(self,x):
        # print(x.shape)
        y = self.dense(x)
        # print(y.shape)
        z = self.self_attention(y)
        #print(z.shape)
        p = z * x
        p = self.softmax(p)
        #print(p.shape)
        A = p * x
        #print(A.shape)
        A = A.reshape(-1,self.k)
        A = self.dropout(A)
        return p,A
"""
sa = self_attention(64,512)
params = 0
for p in sa.parameters():
    if p.requires_grad:
        params += p.numel()
print(params)#16576
"""