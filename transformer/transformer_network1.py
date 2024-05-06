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
        self.wo = nn.Linear(len(self.MHABlock) * out_channels, in_channels)
    
    def forward(self, x):
        z = torch.cat([head(x) for head in self.MHABlock], 1)
        return self.wo(z)


class TransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(TransformerLayer, self).__init__()
        
        self.mha = MultiHeadAttentionBlock(in_channels, out_channels, num_heads)
        self.norm = nn.LayerNorm(in_channels)
        
    def forward(self, x):
        return self.norm(x + self.mha(x))


class Feedforward(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Feedforward, self).__init__()
        
        self.layer1 = nn.Linear(in_channels, hidden_channels)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(hidden_channels, in_channels)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        return self.layer2(x)

        
class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, ff_hidden_channels, out_channels, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        
        self.num_layers = num_layers
        
        self.encoder = nn.Sequential(
            *[TransformerLayer(in_channels, out_channels, num_heads) for _ in range(num_layers)]
        )
        
        self.ff = Feedforward(in_channels, ff_hidden_channels)
    
    def forward(self, x):
        x = self.encoder(x)
        return self.ff(x) + x


class Model(nn.Module):
    def __init__(self,
                 in_channels,
                 ff_hidden_channels,
                 attention_out_channels,
                 num_attention_heads,
                 num_transformer_layers):
        super(Model, self).__init__()
        
        # Embedding layer - nn.Linear
        self.input_embedding = nn.Linear(in_channels, in_channels)
        self.pos_embedding = nn.Linear(in_channels, in_channels)
        
        # Transformer encoder
            # N transformer layers
                # M Multi-head self-attention blocks
                    # Self-attention block
                # Layer norm
                # Residual (o/p + i/p)
            # Feedforward
                # Linear
                # Activation function
                # Linear
        self.encoder = TransformerEncoder(in_channels,
                                          ff_hidden_channels,
                                          attention_out_channels,
                                          num_attention_heads,
                                          num_transformer_layers)
    
    def forward(self, x):
        x = self.input_embedding(x) + self.pos_embedding(x)
        x = self.encoder(x)
        return x
        


if __name__ == '__main__':
    # Configuration
    NUM_SAMPLES = 5
    EMBEDDING_DIM = 10
    KEY_DIM = 256
    FF_HIDDEN_DIM = 64
    QUERY_DIM = 256
    VALUE_DIM = 256
    NUM_HEADS = 4
    NUM_TRANSFORMER_LAYERS = 4

    x = torch.randn((NUM_SAMPLES, EMBEDDING_DIM))
    print(x)

    M = Model(EMBEDDING_DIM, FF_HIDDEN_DIM, KEY_DIM, NUM_HEADS, NUM_TRANSFORMER_LAYERS)
    print(M(x).shape)