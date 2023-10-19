import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import PIL


class PascalVOC2012(Dataset):
    def __init__(self, data_root, image_set) -> None:
        self.dataset = torchvision.datasets.VOCSegmentation(root=data_root, image_set='train', download=False)
        
    def __len__(self):
        return len(self.dataset)
    
    def __transform__(self, image):
        transform_func = transforms.Compose([transforms.Resize((320, 320)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform_func(image)
    
    
    def __get_label_vector__(self, targetpil): #Done with testing
        """From <targetpil> which is a segmentation ground truth, return a
        Python list of 0 or 1 ints (multi-hot vector) representing the classification
        ground truth"""
        classes = set(np.array(targetpil).flatten().tolist())
        label = [0] * 20
        #>>> [x for x in range(1,21)]
        #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        for classnum in range(1,21):
            if classnum in classes:
                #Python is zero-indexed so we have to subtract one
                label[classnum-1] = 1
        return label
    
    
    def __getitem__(self, index) -> dict:
        imagepil, targetpil = self.dataset[index]
        
        targetpil = targetpil.resize((320, 320), resample=PIL.Image.NEAREST)
        target = np.array(targetpil).astype('int').squeeze()
        label = torch.Tensor(self.__get_label_vector__(targetpil))
        
        #convert image to Tensor and normalize
        image = self.__transform__(imagepil) #e.g. out shape torch.Size([3, 500, 334])
        #resample the image to 320 x 320
        # image = image.unsqueeze(0)
        # image = torch.nn.functional.interpolate(image,size=(320, 320),mode='bicubic')
        # image = image.squeeze()
        
        sample = {'data':image, #preprocessed image, for input into NN
                  'seglabel':target,
                  'label':label,
                  'img_idx':index}
        
        return sample



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



def downloadPascalVOC(data_root, batch_size):
    trainloader = torch.utils.data.DataLoader(PascalVOC2012(data_root, 'train'),
                                              shuffle=True,
                                              batch_size=batch_size,
                                              num_workers=8)
        
    testloader = torch.utils.data.DataLoader(PascalVOC2012(data_root, 'val'),
                                             shuffle=True,
                                             batch_size=batch_size,
                                             num_workers=8)
    
    return trainloader, testloader