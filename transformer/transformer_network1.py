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
        x = self.input_embedding(x) + self.pos_embedding(torch.arange(seq_len))
        x = self.encoder(x)
        logits = self.linear(x)
        return logits
        

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


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
        'num_transformer_layers' : 4,
        'dropout_rate' : 0.1,
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
    
    # Evaluate model
    M.eval() # disable dropout
    start_context = "Money's only"

    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=M,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=cfg['context_length']
    )

    print("Output:", out)
    print("Output length:", len(out[0]))
    
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)