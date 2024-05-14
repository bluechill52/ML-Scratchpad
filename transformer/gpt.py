import math
import torch.nn as nn
import torch.nn.functional as F
import torch


class AttentionBlock(nn.Module):
    def __init__(self, cfg):
        super(AttentionBlock, self).__init__()
        
        self.in_channels = cfg['embedding_dim']
        self.out_channels = cfg['qkv_dim']
        self.wk = nn.Linear(self.in_channels, self.out_channels, bias=True)
        self.wq = nn.Linear(self.in_channels, self.out_channels, bias=True)
        self.wv = nn.Linear(self.in_channels, self.out_channels, bias=True)
        
    def forward(self, x):
        K = self.wk(x)
        Q = self.wq(x)
        V = self.wv(x)
        
        # Self-attention matrix - (batch_size x context_length x context_length)
        # Softmax along each row of each batch sample - 3rd dimension of self-attention matrix
        return F.softmax(Q @ K.transpose(1, 2) / math.sqrt(self.out_channels), dim=2) @ V


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, cfg):
        super(MultiHeadAttentionBlock, self).__init__()
        
        self.num_heads = cfg['num_heads']
        self.MHABlock = nn.ModuleList([AttentionBlock(cfg) for _ in range(cfg['num_heads'])])
        self.wo = nn.Linear(len(self.MHABlock) * cfg['qkv_dim'], cfg['embedding_dim'])
    
    def forward(self, x):
        z = torch.cat([head(x) for head in self.MHABlock], 2)
        return self.wo(z)


class Feedforward(nn.Module):
    def __init__(self, cfg):
        super(Feedforward, self).__init__()
        
        self.layer1 = nn.Linear(cfg['embedding_dim'], cfg['ff_hidden_dim'])
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(cfg['ff_hidden_dim'], cfg['embedding_dim'])
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        return self.layer2(x)


class TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super(TransformerLayer, self).__init__()
        
        self.dropout = nn.Dropout(cfg['dropout_rate'])
        self.mha = MultiHeadAttentionBlock(cfg)
        self.norm = nn.LayerNorm(cfg['embedding_dim'])
        self.ff = Feedforward(cfg)
        
    def forward(self, x):
        # First step
        x_ = x
        x = self.norm(x)
        x = self.mha(x)
        x = self.dropout(x)
        
        # Skip connection
        y = x + x_
        
        # Second step
        y_ = y
        y = self.norm(y)
        y = self.ff(y)
        y = self.dropout(y)
        
        # Skip connection
        return y + y_

        
class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(TransformerEncoder, self).__init__()
        
        self.num_layers = cfg['num_transformer_layers']
        
        self.encoder_stack = nn.Sequential(
            *[TransformerLayer(cfg) for _ in range(cfg['num_transformer_layers'])]
        )
    
    def forward(self, x):
        x = self.encoder_stack(x)
        return x


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        
        # Embedding layer - nn.Linear
        self.input_embedding = nn.Embedding(cfg['vocab_size'], cfg['embedding_dim'])
        self.pos_embedding = nn.Embedding(cfg['context_length'], cfg['embedding_dim'])
        
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
        self.encoder = TransformerEncoder(cfg)
        self.linear = nn.Linear(cfg['embedding_dim'], cfg['vocab_size'])
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.input_embedding(x) + self.pos_embedding(torch.arange(seq_len, device=x.device))
        x = self.encoder(x)
        logits = self.linear(x)
        return logits


if __name__ == '__main__':
    pass
