import os
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from datasets import createDataloaderV1, DatasetV1
from tokenizers import SimpleTokenizer


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
        
        return F.softmax(Q @ K.T/ math.sqrt(self.out_channels), dim=1) @ V


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, cfg):
        super(MultiHeadAttentionBlock, self).__init__()
        
        self.num_heads = cfg['num_heads']
        self.MHABlock = nn.ModuleList([AttentionBlock(cfg) for _ in range(cfg['num_heads'])])
        self.wo = nn.Linear(len(self.MHABlock) * cfg['qkv_dim'], cfg['embedding_dim'])
    
    def forward(self, x):
        z = torch.cat([head(x) for head in self.MHABlock], 1)
        return self.wo(z)


class TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super(TransformerLayer, self).__init__()
        
        self.mha = MultiHeadAttentionBlock(cfg)
        self.norm = nn.LayerNorm(cfg['context_length'])
        
    def forward(self, x):
        return self.norm(x + self.mha(x))


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

        
class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(TransformerEncoder, self).__init__()
        
        self.num_layers = cfg['num_transformer_layers']
        
        self.encoder_stack = nn.Sequential(
            *[TransformerLayer(cfg) for _ in range(cfg['num_transformer_layers'])]
        )
        
        self.ff = Feedforward(cfg)
    
    def forward(self, x):
        x = self.encoder_stack(x)
        return self.ff(x) + x


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
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.input_embedding(x) + self.pos_embedding(torch.arange(seq_len))
        
        A = AttentionBlock(cfg)
        print(A(x))
        x = self.encoder(x)
        return x
        


if __name__ == '__main__':
    # Configuration
    cfg = {
        'batch_size' : 4,
        'stride' : 1,
        'context_length' : 5,
        'embedding_dim' : 10,
        'qkv_dim' : 256,
        'ff_hidden_dim' : 64,
        'num_heads' : 4,
        'num_transformer_layers' : 4
    }
    
    
    fname = 'the-verdict.txt'

    tokenizer = SimpleTokenizer(fname)
    with open(os.path.join(os.getcwd(), fname), 'r', encoding='utf-8') as f:
        raw_text = f.read()

    dataset = DatasetV1(raw_text, tokenizer, cfg)
    dataloader = createDataloaderV1(raw_text, tokenizer, cfg)
    
    data_iter = iter(dataloader)
    ip, target = next(data_iter)
    
    cfg['vocab_size'] = tokenizer.getVocabSize()
    
    M = Model(cfg)
    print(M(ip).shape)