import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset


class CaliDataset(Dataset):
    def __init__(self, X, Y) -> None:
        super().__init__()
        
        self.X = torch.reshape(torch.asarray(X), X.shape)
        self.Y = torch.reshape(torch.asarray(Y), Y.shape)
        
    def __getitem__(self, index):
        return {
            'x' : self.X[index],
            'y' : self.Y[index]
        }
    
    def __len__(self):
        return self.X.shape[0]


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.activation = self.relu
        self.fc1 = nn.Linear(8, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 1)
        
        
    def forward(self, x):
        x = self.bn1(self.activation(self.fc1(x)))
        x = self.bn2(self.activation(self.fc2(x)))
        x = self.bn3(self.activation(self.fc3(x)))
        x = self.bn4(self.activation(self.fc4(x)))
        out = self.fc5(x)
        
        return out


if __name__ == '__main__':
    num_epochs = 100
    learning_rate = 0.01
    batch_size = 128
    
    cpu = torch.device('cpu')
    gpu = torch.device('cuda')
    
    norm_data = {}
    df = pd.read_csv('california_housing_train.csv')
    
    # Keep a backup
    df_orig = df.copy(deep=True)
    
    for col in df.columns:
        data = df[col]
        mean = data.mean()
        std = data.std()
        
        norm_data[col] = (mean, std)
        
        df[col] = (data - mean) / std
        
    Y = df['median_house_value'].to_numpy().astype('float32')
    X = df.drop(['median_house_value'], axis=1).to_numpy().astype('float32')
    
    # Initially flatted - convert to 1D
    Y = np.reshape(Y, (Y.shape[0], 1))
    
    loader = torch.utils.data.DataLoader(CaliDataset(X, Y),
                                         shuffle=True,
                                         batch_size=batch_size,
                                         num_workers=8)
    
    model = MLP().to(gpu)
    criterion = nn.MSELoss(reduce='sum')
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_samples = 0
        
        for idx, sample in enumerate(loader):
            x = sample['x'].to(gpu)
            y = sample['y'].to(gpu)
            
            out = model(x)
            
            optim.zero_grad()
            
            loss = criterion(out, y)
            
            loss.backward()
            
            optim.step()
            
            running_loss += loss.item()
            total_samples += x.shape[0]
            
        print(f'*** Epoch {epoch + 1} / {num_epochs} -> total samples {total_samples} avg. loss {running_loss / total_samples}')
        
            
    print('Done')