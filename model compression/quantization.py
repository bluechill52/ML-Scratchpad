import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from models import SimpleNNV1
from torch.optim import SGD
import matplotlib.pyplot as plt



def getAccuracy(model, data):
    # Run model in eval mode to avoid backpropagation
    model.eval()
    
    loader = DataLoader(data, batch_size=500, num_workers=8)
    
    correct, total = 0, 0
    for xs, ts in loader:
        xs = xs.to(device)
        ts = ts.to(device)
        
        zs = model(xs)
        pred = zs.max(1, keepdim=True)[1] # get the index of the max logit
        correct += pred.eq(ts.view_as(pred)).sum().item()
        total += int(ts.shape[0])
        return correct / total


if __name__ == '__main__':
    lr = 0.01
    momentum = 0.9
    epochs = 10
    train_batch_size = 512
    test_batch_size = 512
    device = torch.device('cuda')
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Get MNIST dataset
    train_dataset = MNIST('./data', train=True, transform=transform)
    test_dataset = MNIST('./data', train=False, transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=8)
    
    # Input channels = 28x28 = 784
    M = SimpleNNV1(784).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optim = SGD(M.parameters(), lr=lr, momentum=momentum)
    
    iters = []
    iters_sub = []
    losses = []
    train_acc = []
    test_acc = []
    num_iters = 0
    
    for epoch in range(epochs):
        train_loss = 0.0
        
        # Run model in training mode
        M.train()
        
        for idx, data in enumerate(train_dataloader):
            imgs = data[0].to(device)
            targets = data[1].to(device)
            probs = M(imgs)
            optim.zero_grad()
            loss = criterion(probs, targets)
            loss.backward()
            optim.step()
            train_loss += loss.item()
            
            losses.append(loss)
            iters.append(num_iters)
            
            # Store train and test accuracy every 10 iterations
            if num_iters % 10 == 0:
                iters_sub.append(num_iters)
                train_accuracy = getAccuracy(M, train_dataset)
                test_accuracy = getAccuracy(M, test_dataset)
                
                train_acc.append(train_accuracy)
                test_acc.append(test_accuracy)
                print(f'Iter {num_iters} training accuracy {train_accuracy} test accuracy {test_accuracy}')
                
            num_iters += 1
            
    # plotting
    plt.title(f'Training Curve (batch_size={train_batch_size}, lr={lr})')
    plt.plot(iters, losses, label='Train')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('LossCurve.png')

    plt.title(f'Training Curve (batch_size={test_batch_size}, lr={lr})')
    plt.plot(iters_sub, train_acc, label='Train')
    plt.plot(iters_sub, test_acc, label='Validation')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()
    plt.savefig('AccuracyCurve.png')


    