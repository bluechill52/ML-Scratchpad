import torch
import torchtext
import tiktoken
from torchtext.datasets import IMDB
from collections import defaultdict
from gpt import Model

from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from functools import partial
import torch.nn as nn
import math
import numpy as np



class IMDBDataset(Dataset):
  def __init__(self, ips, labels):
    # import and initialize dataset    
    self.x = np.array(ips, dtype = int)
    self.y = np.array(labels, dtype = int)

  def __getitem__(self, idx):
    # get item by index
    return self.x[idx], self.y[idx]
  
  def __len__(self):
    # returns length of data
    return len(self.x)


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        
        self.ip_embedding = nn.Embedding(cfg['vocab_size'], cfg['embedding_dim'])
        self.pos_embedding = nn.Embedding(cfg['context_length'], cfg['embedding_dim'])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg['embedding_dim'],
            nhead=cfg['num_heads'],
            dim_feedforward=cfg['ff_hidden_dim'],
            dropout=cfg['dropout'],
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg['num_transformer_layers'],
        )
        
        self.classifier = nn.Linear(cfg['embedding_dim'], 2)
        self.d_model = cfg['embedding_dim']
    
    def forward(self, x):
        tokens = self.ip_embedding(x)
        b, t, k = tokens.size()
        
        positions = torch.arange(t, device=x.device)
        positions = self.pos_embedding(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        
        return x



def createImdbDataset(settings, cfg):
    ips = []
    labels = []
    
    # TODO: All labels in train split is positive - hence using test split
    imdb_datapipe = IMDB(root=settings['root'], split='test')
    
    imdb_datapipe = imdb_datapipe.batch(1)
    imdb_datapipe = imdb_datapipe.shuffle()
    imdb_datapipe = imdb_datapipe.rows2columnar(["label", "text"])
    imdb_dataloader = DataLoader(imdb_datapipe, batch_size=None)
    
    tokenizer = tiktoken.get_encoding(settings['encoding'])
    cfg['vocab_size'] = tokenizer.n_vocab
    
    for idx, data in enumerate(imdb_dataloader):
        text = data['text'][0]
        label = data['label'][0]
        tokens = tokenizer.encode(text)
        
        # Store maximum number of tokens as context size
        if len(tokens) > cfg['context_length']:
            tokens = tokens[:cfg['context_length']]
        else:
            for _ in range(cfg['context_length'] - len(tokens)):
                # TODO: Add pad token - right now padding with '#' char (0)
                tokens.append(0)
                
        ips.append(tokens)
        labels.append(label - 1)
        
    return ips, labels
    
        
    
if __name__ == '__main__':
    cfg = {
        'stride' : 1,
        'context_length' : 50,
        'embedding_dim' : 128,
        'qkv_dim' : 256,
        'ff_hidden_dim' : 64,
        'num_heads' : 8,
        'num_transformer_layers' : 4,
        'dropout' : 0.1,
    }
    
    settings = {
        'root' : '../data',
        'batch_size' : 16,
        'train_split' : 0.8,
        'encoding' : 'gpt2',
        'num_epochs' : 100,
        'lr' : 1e-4,
        'weight_decay' : 0.2,
        'device' : 'cuda',
        'eval_freq' : 50
    }
    
    ips, labels = createImdbDataset(settings, cfg)
    
    dataset = IMDBDataset(ips, labels)
    
    num_samples = len(dataset)
    test_split = 1 - settings['train_split']
    test_size = int(num_samples * test_split)

    indices = list(range(num_samples))
    
    test_idx = np.random.choice(indices, size = test_size, replace = False)
    train_idx = list(set(indices) - set(test_idx))
    
    train_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)
    
    train_dataloader = DataLoader(dataset, batch_size=settings['batch_size'], num_workers=8, sampler=train_sampler)
    test_dataloader = DataLoader(dataset, batch_size=settings['batch_size'], num_workers=8, sampler=test_sampler)
    
    device = torch.device(settings['device'])
    
    torch.manual_seed(123)
    
    # model = Net(cfg).to(device)
    model = Model(cfg).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=settings['lr'])
    
    for epoch in range(settings['num_epochs']):
        train_epoch_loss = 0
        train_epoch_samples = 0
        train_epoch_correct = 0
        for idx, data in enumerate(train_dataloader):
            ip = data[0].to(device)
            labels = data[1].to(device)
            
            optim.zero_grad()
            
            predictions = model(ip)
            
            loss = criterion(predictions, labels)
            
            train_epoch_loss += loss.item()
            
            correct = predictions.argmax(axis=1) == labels
            
            train_epoch_correct += correct.sum().item()
            train_epoch_samples += correct.size(0)
            
            
            loss.backward()
            optim.step()
        
        print(f'Epoch {epoch} train loss {train_epoch_loss} train accuracy {(train_epoch_correct / train_epoch_samples) * 100.0}')
        
        # Check accuracy after this epoch
        with torch.no_grad():
            test_epoch_loss = 0
            test_epoch_samples = 0
            test_epoch_correct = 0

            for idx, data in enumerate(test_dataloader):
                predictions = model(data[0].to(device))
                labels = data[1].to(device)
                
                test_loss = criterion(predictions, labels)
                
                test_epoch_loss += test_loss.item()
                
                correct = predictions.argmax(axis=1) == labels
                
                test_epoch_correct += correct.sum().item()
                test_epoch_samples += correct.size(0)
                
            print(f'Epoch {epoch} test loss {test_epoch_loss} test accuracy {(test_epoch_correct / test_epoch_samples) * 100.0}')
            
        
        