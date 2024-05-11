import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler
from models import LeNet, SmallNet, AlexNet
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


def compute_accuracy(model, loader, device):
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
    
    
def downloadCIFAR10(data_root, batch_size):
    train_transform = transforms.Compose([transforms.Resize((70, 70)),
                                          transforms.RandomCrop((64, 64)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    test_transform = transforms.Compose([transforms.Resize((64, 64)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                            download=True, transform=train_transform)
    
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                           download=True, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=batch_size,
                                              num_workers=8,
                                              shuffle=True,
                                              drop_last=True)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8)

    return trainloader, testloader


def downloadMNIST(data_root, batch_size):
    trainset = torchvision.datasets.MNIST(root=data_root, train=True,
                                            download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=8)

    testset = torchvision.datasets.MNIST(root=data_root, train=False,
                                        download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=8)

    return trainloader, testloader


def train(net, trainloader, validloader, num_epochs, learning_rate, momentum, device, tensorboard_writer):
    print(f'Total number of parameters in network: {sum(p.numel() for p in net.parameters())}')
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
    
    loss_vals = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # Iterate over each batch
        # Set model to train
        net.train()
        
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
            
            '''
            if idx % 20 == 19:
                # ...log the running loss
                tensorboard_writer.add_scalar('training loss',
                                running_loss / 20,
                                epoch * len(trainloader) + idx)

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch
                tensorboard_writer.add_figure('predictions vs. actuals',
                                plot_classes_preds(net, features, labels),
                                global_step=epoch * len(trainloader) + idx)
                
                running_loss = 0.0
            '''
                
        
        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch + 1} / {num_epochs}, Loss: {epoch_loss}")
        
        # Compute accuracy after each epoch
        # Set model to eval
        net.eval()
        
        train_accuracy = compute_accuracy(net, trainloader, device)
        test_accuracy = compute_accuracy(net, testloader, device)
        print(f'Epoch {epoch + 1} finished --> train accuracy {train_accuracy} validation accuracy {test_accuracy}')
        
        loss_vals.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
    params = {
        'Model' : net,
        'Epoch_losses' : loss_vals,
        'Train_accuracies' : train_accuracies,
        'Test_accuracies' : test_accuracies
    }
    
    return params


def train_kd(teacher_model, student_model, trainloader, testloader, num_epochs, learning_rate, momentum, device):
    print(f'Total number of parameters in teacher network: {sum(p.numel() for p in teacher_model.parameters())}')
    print(f'Total number of parameters in student network: {sum(p.numel() for p in student_model.parameters())}')
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(student_model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    
    loss_vals = []
    train_accuracies = []
    test_accuracies = []
    
    teacher_model.eval()
    student_model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # Iterate over each batch
        for idx, data in enumerate(trainloader):
            features, labels = data[0].to(device), data[1].to(device)
            
            # Initialize optimizer
            optimizer.zero_grad()
            
            # Forward pass features through net to get teacher predictions
            with torch.no_grad():
                teacher_preds = teacher_model(features)
            
            # Forward pass features through net to get student predictions
            student_preds = student_model(features)
            
            soft_targets = nn.functional.softmax(teacher_preds / 2, dim=-1)
            soft_preds = nn.functional.softmax(student_preds / 2, dim=-1)
            
            soft_loss = torch.sum(soft_targets * soft_preds) / soft_preds.size()[0] * 4
            label_loss = criterion(soft_preds, labels)
            
            loss = 0.25 * soft_loss + 0.75 * label_loss
            
            # Backpropagate loss through net
            loss.backward()
            
            # Step through optimizer
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch + 1} / {num_epochs}, Loss: {epoch_loss}")
        
        # Compute accuracy after each epoch
        train_accuracy = compute_accuracy(student_model, trainloader, device)
        test_accuracy = compute_accuracy(student_model, testloader, device)
        print(f'Epoch {epoch + 1} finished --> train accuracy {train_accuracy} test accuracy {test_accuracy}')
        
        loss_vals.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
    params = {
        'Model' : student_model,
        'Epoch_losses' : loss_vals,
        'Train_accuracies' : train_accuracies,
        'Test_accuracies' : test_accuracies
    }
    
    return params


#### helper functions

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    img = img.to(torch.device('cpu'))
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.to(torch.device('cpu')).numpy())
    return preds, [nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)


if __name__ == '__main__':
    # Download dataset
    # Create train, validation and test splits and batches
    # Initialize model
    # Initialize loss function
    # Initialize optimizer
    # Loop over each batch
    
    random_seed = 1
    data_root = './data'
    batch_size = 64
    num_epochs = 200
    learning_rate = 0.0001
    momentum = 0.9
    tensorboard_log_dir = 'runs'
    run_suffix = 'cifar_10_experiments'
    
    set_all_seeds(random_seed)
    
    classes = ('plane', 'car', 'bird', 'cat', 
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    num_classes = len(classes)
    
    tensorboard_writer = SummaryWriter(os.path.join(tensorboard_log_dir, run_suffix))
    
    device = torch.device('cuda')
    
    trainloader, testloader = downloadCIFAR10(data_root, batch_size)
    # trainloader, testloader = downloadMNIST(data_root, batch_size)

    # Train big network
    # lenet = LeNet().to(device)
    
    torch.manual_seed(random_seed)
    
    alexnet = AlexNet(num_classes).to(device)
    
    alexnet_train_info = train(alexnet, trainloader, testloader, num_epochs, learning_rate, momentum, device, tensorboard_writer)
    
    # Optionally save trained model weights
    small_net = SmallNet().to(device)
    
    # Create another copy small_net to train using KD
    small_net_kd = SmallNet().to(device)
    
    small_train_info = train(small_net, trainloader, testloader, num_epochs, learning_rate, momentum, device)
    print(f'Total number of parameters in network: {sum(p.numel() for p in small_net.parameters())}')
    
    kd_train_info = train_kd(alexnet, small_net_kd, trainloader, testloader, num_epochs, learning_rate, momentum, device)
    
    print('Done')