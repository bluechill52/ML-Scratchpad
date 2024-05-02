import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AttentionBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.wk = nn.Linear(in_dim, out_dim)
        self.wq = nn.Linear(in_dim, out_dim)
        
        # Assume dimension of key, queries and values are same
        # Although dimension of values could be different - add a linear
        # projection layer to make it same as key and queries - since output
        # will be input to next layer
        self.wv = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        K = self.wk(x)
        Q = self.wq(x)
        V = self.wv(x)
        
        S = F.softmax(Q @ K.T, dim=1)
        return S @ V


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.heads = nn.ModuleList([AttentionBlock(in_dim, out_dim) for _ in range(num_heads)])
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads])
        return nn.Dropout(nn.Linear(out, out_dim, out_dim))


class FeedForward(nn.Module):
    def __init__(self, in_dim):
        super(FeedForward, self).__init__()
        
        # Keep input and output dimensions same
        self.net = nn.Sequential(
            nn.Linear(in_dim, 4 * in_dim),
            nn.ReLU(),
            nn.Linear(4 * in_dim, in_dim),
            nn.Dropout(),
            nn.LayerNorm(in_dim),
        )
        
    def forward(self, x):
        return self.net(x)
        
        
class TransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(TransformerLayer, self).__init__()
        self.MHABlock = MultiHeadAttentionBlock(in_dim, out_dim, num_heads)
        self.ffn = FeedForward(out_dim)
        self.layerNorm = nn.LayerNorm(out_dim)
        
    def forward(self, x):
        x = self.layerNorm(x + self.MHABlock(x))
        x = self.layerNorm(x + self.ffn(x))
        return x


class Model(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, num_heads):
        super(Model, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.net = nn.ModuleList([TLayer(in_dim, out_dim, num_heads) for _ in range(num_layers)])
    
    def forward(self, x):
        return self.net(x)
    
    
def run():
    pass

if __name__ == '__main__':
    # Load token dataset
    
    run()