import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms



class AlexNet(nn.Sequential):
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


def compute_accuracy(model, loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
    
    
def download():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    batch_size = 128

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    epochs = 10
    
    device = torch.device('cuda')
    
    net = AlexNet().to(device)
    
    print(f'Total number of parameters in network: {sum(p.numel() for p in net.parameters())}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    loss_vals = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for idx, data in enumerate(trainloader):
            features, labels = data[0].to(device), data[1].to(device)
            
            preds = net(features)
            
            optimizer.zero_grad()
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch + 1} / {epochs}, Loss: {epoch_loss}")
        
        # Compute accuracy after each epoch
        train_accuracy = compute_accuracy(net, trainloader, device)
        test_accuracy = compute_accuracy(net, testloader, device)
        print(f'Epoch {epoch + 1} finished --> train accuracy {train_accuracy} test accuracy {test_accuracy}')
        
        loss_vals.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
    print('Done')
    
    
if __name__ == '__main__':
    # Download dataset
    # Create train, validation and test splits and batches
    # Initialize model
    # Initialize loss function
    # Initialize optimizer
    # Loop over each batch
    
    download()