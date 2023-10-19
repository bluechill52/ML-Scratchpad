import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from helper_data import downloadPascalVOC


if __name__ == '__main__':
    data_root = './data'
    batch_size = 128
    
    trainloader, testloader = downloadPascalVOC(data_root, batch_size)
    
    for idx, sample in enumerate(trainloader):
        print(sample)
        
    print('Done')
