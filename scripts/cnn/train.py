# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
from ai.cnn import ConvNeuralNet

# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20
data_path = './data/'
os.makedirs(data_path, exist_ok=True)
model_path = f'{data_path}/models/cnn'
os.makedirs(f'{model_path}', exist_ok=True)
# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Selected device: [{device}]')

# Use transforms.compose method to reformat images for modeling,
# and save to variable all_transforms for later use
all_transforms = transforms.Compose([transforms.Resize((32,32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.2023, 0.1994, 0.2010])
                                     ])
#TODO: replace mean and std with live computation

# Create Training dataset
train_dataset = torchvision.datasets.CIFAR10(root = data_path,
                                             train = True,
                                             transform = all_transforms,
                                             download = True)



# Instantiate loader objects to facilitate processing
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)



model = ConvNeuralNet(num_classes)
# Set Loss function with criterion
criterion = nn.CrossEntropyLoss()

# Set optimizer with optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

total_step = len(train_loader)

# We use the pre-defined number of epochs to determine how many iterations to train the network on
loss_total = np.zeros((num_epochs,))
for epoch in range(num_epochs):
	#Load in the data in batches using the train_loader object
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }
            , f'{model_path}/checkpoint_{epoch:04d}.pt')

    if epoch == 0 or loss < loss_total.min():
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }
                , f'{model_path}/checkpoint_best.pt')
    
        model_scripted = torch.jit.script(model)
        model_scripted.save(f'{model_path}/model_best.ts')
                
    loss_total[epoch] = loss
    
