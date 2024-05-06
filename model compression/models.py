import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets import MNIST


class SimpleNNV1(nn.Module):
    def __init__(self, in_channels):
        super(SimpleNNV1, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        logits = self.model(x)
        return logits


if __name__ == '__main__':
    x = torch.randn((1, 1, 28, 28))
    M = SimpleNNV1(784)
    print(M(x))