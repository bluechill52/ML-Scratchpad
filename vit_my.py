import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tqdm


class AttentionBlock(nn.Module):
    def __init__(self, in_dim, head_dim=20):
        super(AttentionBlock, self).__init__()
        self.head_dim = 20
        self.wk = nn.Linear(in_dim, head_dim)
        self.wq = nn.Linear(in_dim, head_dim)
        self.wv = nn.Linear(in_dim, head_dim)
        
    def forward(self, x):
        # Size of x - N * d
        K = self.wk(x)
        Q = self.wq(x)
        V = self.wv(x)
        
        D = nn.functional.softmax(K @ Q.T / self.head_dim ** 0.5, dim=1)
        return D @ V
        
        
        
class MSA(nn.Module):
    def __init__(self, in_dim, head_dim=20, num_heads=5):
        super(MSA, self).__init__()
        self.attentionHeads = nn.ModuleList([AttentionBlock(in_dim, head_dim) for _ in range(num_heads)])
        
    def forward(self, x):
        return self.attentionHeads(x)



if __name__ == '__main__':
    x = torch.randn([5, 20])
    print(x)
    A = AttentionBlock(x.shape[1])
    print(A(x))
    
    M = MSA(x.shape[1])
    print(MSA(x))
    print('Done')