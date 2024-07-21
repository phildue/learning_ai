# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from ai.cnn import ConvNeuralNet

# Define relevant variables for the ML task
batch_size = 64
learning_rate = 0.0001
num_epochs = 20
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_classes = len(classes)
checkpoint = f'epoch-20-lr-{0.001:.6f}-bs-{batch_size}/checkpoint_best.pt'
experiment_name = f'epoch-{num_epochs}-lr-{learning_rate:.6f}-bs-{batch_size}-checkpoint-{checkpoint[:-3]}'
data_path = './data/'
model_path = f'{data_path}/models/cnn/{experiment_name}'
os.makedirs(data_path, exist_ok=True)
os.makedirs(f'{model_path}', exist_ok=True)

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

train_dataset = torchvision.datasets.CIFAR10(root = data_path,
                                             train = True,
                                             transform = all_transforms,
                                             download = True)



train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)


model = ConvNeuralNet(num_classes)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  
if checkpoint is not None:
    checkpoint = torch.load(f'{data_path}/models/cnn/{checkpoint}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
total_step = len(train_loader)

loss_total = np.zeros((num_epochs,))+np.inf
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(tqdm(train_loader)):  
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    if epoch % 10 == 0 or epoch == num_epochs-1:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }
                , f'{model_path}/checkpoint_{epoch:04d}.pt')

    if epoch == 0 or loss < loss_total.min():
        print(f'Saving new best model at epoch [{epoch}] with loss [{loss}] < [{loss_total.min()}] to [{model_path}/checkpoint_best.pt] and [{model_path}/model_best.ts]')
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
    
print(f'Training completed after [{epoch}] epochs. Minimal loss: [{loss_total.min()}]')