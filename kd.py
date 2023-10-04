import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from models import CustomNet


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
    
    
def download(data_root, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    return trainloader, testloader


def train(trainloader, testloader, num_epochs, learning_rate, momentum):    
    device = torch.device('cuda')
    
    net = CustomNet().to(device)
    
    print(f'Total number of parameters in network: {sum(p.numel() for p in net.parameters())}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    
    loss_vals = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # Iterate over each batch
        for idx, data in enumerate(trainloader):
            features, labels = data[0].to(device), data[1].to(device)
            
            # Forward pass features through net to get predictions
            preds = net(features)
            
            # Initialize optimizer
            optimizer.zero_grad()
            
            # Get loss from criterion based on predictions and actual labels
            loss = criterion(preds, labels)
            
            # Backpropagate loss through net
            loss.backward()
            
            # Step through optimizer
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch + 1} / {num_epochs}, Loss: {epoch_loss}")
        
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
    
    data_root = './data'
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001
    momentum = 0.9
    
    classes = ('plane', 'car', 'bird', 'cat', 
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    trainloader, testloader = download(data_root, batch_size)
    
    train(trainloader, testloader, num_epochs, learning_rate, momentum)
    