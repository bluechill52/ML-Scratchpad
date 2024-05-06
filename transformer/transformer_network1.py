import math
import torch.nn as nn
import torch.nn.functional as F
import torch


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wk = nn.Linear(in_channels, out_channels, bias=True)
        self.wq = nn.Linear(in_channels, out_channels, bias=True)
        self.wv = nn.Linear(in_channels, out_channels, bias=True)
        
    def forward(self, x):
        K = self.wk(x)
        Q = self.wq(x)
        V = self.wv(x)
        
        return F.softmax(Q @ K.T/ math.sqrt(self.out_channels), dim=1) @ V


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        
        self.num_heads = num_heads
        self.MHABlock = nn.ModuleList([AttentionBlock(in_channels, out_channels) for _ in range(num_heads)])
        self.wo = nn.Linear(out_channels, in_channels)
    
    def forward(self, x):
        z = torch.cat([head(x) for head in self.MHABlock], 1)
        return self.wo(z)


class TransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(TransformerLayer, self).__init__()
        
        self.mha = MultiHeadAttentionBlock(in_channels, out_channels, num_heads)
        self.norm = nn.LayerNorm(in_channels)
        
    def forward(self, x):
        return self.norm(x + mha(x))
    

NUM_SAMPLES = 5
EMBEDDING_DIM = 10
KEY_DIM = 256
QUERY_DIM = 256
VALUE_DIM = 256
NUM_HEADS = 4

x = torch.randn((NUM_SAMPLES, EMBEDDING_DIM))
m = MultiHeadAttentionBlock(EMBEDDING_DIM, KEY_DIM, NUM_HEADS)
print(x)
print(m(x).shape)