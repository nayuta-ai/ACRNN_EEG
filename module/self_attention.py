import torch
import torch.nn as nn
import numpy as np

class dense(nn.Module):
    def __init__(self,input_dim1,input_dim2,output_dim,
    activation=lambda x:x):
        super().__init__()
        self.W1 = nn.Parameter(torch.Tensor(np.random.normal(size=(input_dim1,output_dim))))
        self.W2 = nn.Parameter(torch.Tensor(np.random.normal(size=(input_dim2,output_dim))))
        self.b = nn.Parameter(torch.Tensor(np.zeros(output_dim)))
        self.activation = activation
        self.vector = nn.Linear(input_dim2,input_dim2)
    def forward(self,x):
        y = self.vector(x)
        return self.activation(torch.matmul(x,self.W1)+torch.matmul(y,self.W2)+self.b)

class self_attention(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.q = input_dim
        self.k = input_dim
        self.output = output_dim
        self.dense = dense(self.q,self.k,self.output)
        self.self_attention = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.output,self.k)
        )
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self,x):
        print(x.shape)
        y = self.dense(x)
        print(y.shape)
        z = self.self_attention(y)
        print(z.shape)
        p = z.transpose(1,2) * x
        p = self.softmax(p)
        print(p.shape)
        A = p * x
        print(A.shape)
        return A

input_dim = 64
output_dim = 512
model = self_attention(64,512)
a = torch.randn(800,1,64)
b = model(a)
print(b.shape)