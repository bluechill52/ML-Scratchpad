import os
import shutil
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt



class LDCData(Dataset):
    def __init__(self, X, Y) -> None:
        super().__init__()
        
        self.X = torch.reshape(X, (len(X), 1))
        self.Y = torch.reshape(Y, (len(Y), 1))
        
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
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)
        
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        out = self.fc4(x)
        
        return out
        

def gen_data(num_samples):
    # get noise around y observations
    yNormal = torch.distributions.Normal(loc=0.0, scale=10)
    yNoise  = yNormal.sample([num_samples])

    # get observations
    xObs = 10 * torch.rand([num_samples]) - 5    # uniform from [-5,5]
    yObs = xObs ** 3 - xObs ** 2 + 25 * torch.sin(2 * xObs) + yNoise

    return xObs, yObs


def func(X, params):    
    return params['a1'] + X * params['a2'] + X ** 2 * params['a3']


def create_data(num_samples, params):
    xObs = 10 * torch.rand([num_samples]) - 5    # uniform from [-5,5]
    yObs = func(xObs, params)
    
    return xObs, yObs


def run():
    pass
    
if __name__ == '__main__':
    num_samples = 100
    batch_size = 128
    num_epochs = 1000
    learning_rate = 1e-3
    momentum = 0.9
    tensorboard_log_dir = 'runs/non_linear_reg'
    
    params = {
        'a1' : 1.0,
	    'a2' : -0.0193,
	    'a3' : 0.2073,
	    # 'a4' : 0.1755,
	    # 'a5' : 0.0239, 
	    # 'a6' : -0.6065,
	    # 'a7' : 0.6264
    }
    
    # Remove folder contents to refresh tensorboard
    if os.path.isdir(tensorboard_log_dir):
        shutil.rmtree(tensorboard_log_dir)
        
    writer = SummaryWriter('runs/non_linear_reg')

    X, Y = create_data(num_samples, params)
    # X, Y = gen_data(num_samples)
    
    loader = torch.utils.data.DataLoader(LDCData(X, Y),
                                         shuffle=True,
                                         batch_size=batch_size,
                                         num_workers=8)
    
    
    cpu = torch.device('cpu')
    gpu = torch.device('cuda')
    
    model = MLP().to(gpu)
    criterion = nn.MSELoss(reduction='sum')
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_samples = 0
        for idx, sample in enumerate(loader):
            x = sample['x'].to(gpu)
            y = sample['y'].to(gpu)
            
            out = model(x)
            
            optimizer.zero_grad()
            
            loss = criterion(out, y)
            
            running_loss += loss.item()
            total_samples += x.shape[0]
            
            loss.backward()
            
            optimizer.step()
        
        '''
        # Log epoch loss on tensorboard    
        writer.add_scalar('Training loss', running_loss / total_samples, epoch)
        
        # Log predicted outputs and overlay with expected outputs at the end of each epoch
        y_pred = np.array([model(torch.tensor([x]).to(gpu)).detach().to(cpu).numpy() for x in X]).flatten()
        fig = plt.figure()
        plt.scatter(X, Y)
        plt.scatter(X, y_pred)
        
        writer.add_figure('Predictions vs. Actuals', fig, global_step=epoch)
        '''
            
        print(f'*** Epoch {epoch + 1} / {num_epochs} - total samples {total_samples} avg. loss {running_loss / total_samples}')
        
    run()
