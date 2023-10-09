import torch
import torch.nn as nn


class CustomNet(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.bn1(self.pool(self.relu(self.conv1(x))))
        x = self.bn2(self.pool(self.relu(self.conv2(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.bn3(self.fc1(x))
        x = self.bn4(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class SmallNet(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(6 * 14 * 14, 10)
        
    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
