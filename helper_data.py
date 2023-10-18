import torch
import torchvision
from torchvision.transforms import transforms



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