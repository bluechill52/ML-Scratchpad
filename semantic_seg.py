import os
import torch
import torch.nn as nn
import torchvision
from models import UNet, UNetBatchNorm
import matplotlib.pyplot as plt
import numpy as np
from helper_data import downloadPascalVOC
from time import time


if __name__ == '__main__':
    data_root = './data'
    batch_size = 32
    learning_rate = 0.001
    momentum = 0.9
    num_epochs = 100
    continue_from_epoch = 0
    model_path = 'unet_epoch_99.pt'
    
    gpu = torch.device('cuda')
    cpu = torch.device('cpu')
    
    trainloader, testloader = downloadPascalVOC(data_root, batch_size)
    
    # model = UNet(num_classes=21).to(gpu)
    model = UNetBatchNorm(num_classes=21).to(gpu)
    
    model_path = os.path.join(os.getcwd(), model_path)
    if continue_from_epoch > 0 and os.path.exists(model_path):
        print(f'Loading from pretrained model exists --> {model_path}')
        model = torch.load(model_path)
    
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(continue_from_epoch, num_epochs):
        total_loss = 0.0
        total_size = 0
        total_time = 0.0
        for idx, sample in enumerate(trainloader):
            tic = time()
            
            img = sample['data'].to(gpu)
            seglabel = sample['seglabel'].to(gpu)
            
            # Initialize optimizer
            optimizer.zero_grad()
            
            logits = model(img)
            
            # loss = nn.functional.cross_entropy(logits, seglabel, reduction='sum')
            loss = criterion(logits, seglabel)
            
            # Backpropagate
            loss.backward()
            
            # Step through the optimizer
            optimizer.step()
            
            # Accumulate batch loss
            # total_loss += loss
            # total_size += (seglabel.shape[0] * seglabel.shape[1] * seglabel.shape[2])
            total_loss += loss.item()
            
            toc = time()
            total_time += (toc - tic)
            
            # Print time taken every 10 batches
            if idx % 10 == 0:
                print(f'Batch {idx} -> time taken {(toc - tic)}')
            
        avg_time = total_time / len(trainloader)
        avg_loss = total_loss / len(trainloader)
        
        print(f'*** Epoch {epoch} total time taken {total_time} avg. time taken per batch = {avg_time} loss {avg_loss}')
        print('Done')
        
