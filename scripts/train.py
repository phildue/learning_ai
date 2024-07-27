# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from ai.models.vision_transformer import VisionTransformer
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2 as transforms

from ai.models.convolutional_neural_network import ConvNeuralNet

def train(data_loader: torch.utils.data.DataLoader, criterion, optimizer:torch.optim.Optimizer):
    for i, (images, labels) in enumerate(tqdm(data_loader,desc='Training')):  
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss


def evaluate(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module, criterion):
    with torch.no_grad():
        correct = 0
        total = 0
        loss_val = 0
        labels_all = []
        for images, labels in tqdm(data_loader,desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss_val += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            labels_all += labels

    return correct/total, loss_val/len(data_loader), labels_all
        

batch_size = 128
learning_rate = 0.001
num_epochs = 20
model_name = 'cnn'
input_shape = 32,32

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_classes = len(classes)

models = {
    'cnn': ConvNeuralNet(num_classes),
    'vit': VisionTransformer((3, input_shape[0], input_shape[1]), n_patches=8, n_blocks=2, hidden_d=256, n_heads=4, mlp_dim=128, out_d=num_classes)
}


checkpoint = None#f'epoch-20-lr-{0.001:.6f}-bs-{batch_size}/checkpoint_best.pt'
experiment_name = f'epoch-{num_epochs}-lr-{learning_rate:.6f}-bs-{batch_size}'
data_path = './data/'
experiment_path = f'{data_path}/models/{model_name}_{experiment_name}'
params =  {k: eval(k) for k in ('batch_size', 'learning_rate', 'num_epochs','checkpoint','experiment_path')}

writer = SummaryWriter(experiment_path)

os.makedirs(data_path, exist_ok=True)
os.makedirs(f'{experiment_path}', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Selected device: [{device}]')

# Use transforms.compose method to reformat images for modeling,
# and save to variable all_transforms for later use
transform_train = transforms.Compose([
    transforms.RandomCrop(input_shape, padding=4),
    transforms.Resize(input_shape),
    transforms.RandomHorizontalFlip(),
    transforms.ToImage(),
    transforms.ToDtype(torch.float64,scale=True),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(input_shape),
    transforms.ToImage(),
    transforms.ToDtype(torch.float64,scale=True),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
#TODO: replace mean and std with live computation

train_dataset = torchvision.datasets.CIFAR10(root = data_path,
                                             train = True,
                                             transform = transform_train,
                                             download = True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

test_dataset = torchvision.datasets.CIFAR10(root = data_path,
                                            train = False,
                                            transform = transform_test,
                                            download=True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)



model = models[model_name]
model = model.to(device=device, dtype=float)
#writer.add_graph(model,next(iter(train_loader))[0])

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
if checkpoint is not None:
    checkpoint = torch.load(f'{data_path}/models/{model_name}/{checkpoint}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
total_step = len(train_loader)

loss_total = np.zeros((num_epochs,))+np.inf
for epoch in range(num_epochs):
    
    loss_train = train(train_loader, criterion, optimizer)

    writer.add_scalar('loss/train', loss_train, epoch)

    if epoch % 10 == 0 or epoch == num_epochs-1:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_train,
                },f'{experiment_path}/checkpoint_{epoch:04d}.pt')

    print(f'Epoch [{epoch+1}/{num_epochs}], train_loss: {loss_train:.4f}')
    accuracy_val, loss_val, labels = evaluate(test_loader, model, criterion)
    print(f'Epoch [{epoch+1}/{num_epochs}], train_loss: {loss_train:.4f}, val_loss: {loss_val:.4f}, val_accuracy: {accuracy_val:.4f}')

    writer.add_scalar('loss/val', loss_val, epoch)
    writer.add_scalar('accuracy/val', accuracy_val, epoch)

    if epoch == 0 or loss_val < loss_total.min():
        print(f'Saving new best model at epoch [{epoch}] with val_loss [{loss_val}] < [{loss_total.min()}] to [{experiment_path}]')
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_train,
                }, f'{experiment_path}/checkpoint_best.pt')
    
        #torch.jit.script(model).save(f'{experiment_path}/model_best.ts')
    

    writer.flush()
    
    loss_total[epoch] = loss_val
writer.close()            
print(f'Training completed after [{epoch}] epochs. Minimal loss: [{loss_total.min()}]')