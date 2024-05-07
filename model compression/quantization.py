import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from models import SimpleNNV1
from torch.optim import SGD
import matplotlib.pyplot as plt



def getAccuracy(cfg, model, data):
    device = torch.device(cfg['device'])
    
    # Run model in eval mode to avoid backpropagation
    model.eval()
    
    loader = DataLoader(data, batch_size=512, num_workers=8)
    
    correct, total = 0, 0
    for xs, ts in loader:
        xs = xs.to(device)
        ts = ts.to(device)
        
        zs = model(xs)
        pred = zs.max(1, keepdim=True)[1] # get the index of the max logit
        correct += pred.eq(ts.view_as(pred)).sum().item()
        total += int(ts.shape[0])
        return correct / total


def createDataloader(cfg):
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Get MNIST dataset
    train_dataset = MNIST(cfg['dataset_root'], train=True, transform=transform)
    test_dataset = MNIST(cfg['dataset_root'], train=False, transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['train_batch_size'], shuffle=cfg['dataloader_shuffle'], num_workers=cfg['dataloader_num_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['test_batch_size'], shuffle=cfg['dataloader_shuffle'], num_workers=cfg['dataloader_num_workers'])
    
    return {
        'train_dataloader' : train_dataloader,
        'test_dataloader' : test_dataloader,
        'train_dataset' : train_dataset,
        'test_dataset' : test_dataset
    }


def plot(epoch_data):
    lr = epoch_data['lr']
    iters = epoch_data['iters']
    iters_sub = epoch_data['iters_sub']
    losses = epoch_data['losses']
    train_batch_size = epoch_data['train_batch_size'],
    test_batch_size = epoch_data['test_batch_size'],
    train_acc = epoch_data['train_acc']
    test_acc = epoch_data['test_acc']
    
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


def train(cfg, data, model, criterion, optim):
    iters = []
    iters_sub = []
    losses = []
    train_acc = []
    test_acc = []
    num_iters = 0
    
    device = torch.device(cfg['device'])
    
    train_dataloader, test_dataloader = data['train_dataloader'], data['test_dataloader']
    train_dataset, test_dataset = data['train_dataset'], data['test_dataset']
    
    # Load model to device
    model = model.to(device)
    
    for epoch in range(cfg['epochs']):
        train_loss = 0.0
        
        # Run model in training mode
        M.train()
        
        for idx, data in enumerate(train_dataloader):
            imgs = data[0].to(device)
            targets = data[1].to(device)
            probs = model(imgs)
            optim.zero_grad()
            loss = criterion(probs, targets)
            loss.backward()
            optim.step()
            train_loss += loss.item()
            
            losses.append(loss.item())
            iters.append(num_iters)
            
            # Store train and test accuracy every 10 iterations
            if num_iters % 10 == 0:
                iters_sub.append(num_iters)
                train_accuracy = getAccuracy(cfg, model, train_dataset)
                test_accuracy = getAccuracy(cfg, model, test_dataset)
                
                train_acc.append(train_accuracy)
                test_acc.append(test_accuracy)
                print(f'Iter {num_iters} training accuracy {train_accuracy} test accuracy {test_accuracy}')
                
            num_iters += 1
    
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                }, 'trained_model.pt')
    
    epoch_data = {
        'lr' : cfg['lr'],
        'iters' : iters,
        'iters_sub' : iters_sub,
        'losses' : losses,
        'train_batch_size' : cfg['train_batch_size'],
        'test_batch_size' : cfg['test_batch_size'],
        'train_acc' : train_acc,
        'test_acc' : test_acc,
    }
    
    return epoch_data


if __name__ == '__main__':
    cfg = {
        'dataset_root' : './data',
        'lr' : 0.01,
        'momentum' : 0.9,
        'epochs' : 10,
        'dataloader_num_workers' : 8,
        'dataloader_shuffle' : True,
        'train_batch_size' : 512,
        'test_batch_size' : 512,
        'device' : 'cuda',
    }
    
    data = createDataloader(cfg)
    
    # Input channels = 28x28 = 784
    M = SimpleNNV1(784)
    
    criterion = nn.CrossEntropyLoss()
    optim = SGD(M.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
    
    epoch_data = train(cfg, data, M, criterion, optim)
    
    plot(epoch_data)
    