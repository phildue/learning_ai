# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random

from ai.experiment_tracking.tracker import Tracker
from ai.datasets.cifar10 import Cifar10
from ai.models import models
SEED=666
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(10)


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
            labels_all += [predicted]

    return correct/total, loss_val/len(data_loader), labels_all
        

if __name__ == '__main__':
    batch_size = 128
    learning_rate = 0.0001
    num_epochs = 100
    model_name = 'cnn'
    dataset_name = 'cifar10'
    input_shape = 32,32
    data_path = './data/'

    datasets = {
        'cifar10': Cifar10
    }
    dataset = datasets['cifar10'](shape=input_shape, batch_size=batch_size, data_path=data_path)
    

    

    checkpoint = None#f'epoch-20-lr-{0.001:.6f}-bs-{batch_size}/checkpoint_best.pt'
    experiment_name = f'epoch-{num_epochs}-lr-{learning_rate:.6f}-bs-{batch_size}'
    experiment_path = f'{data_path}/models/{model_name}_{experiment_name}'
    params =  {k: eval(k) for k in ('batch_size', 'learning_rate', 'num_epochs','checkpoint','experiment_path','model_name','dataset_name')}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Selected device: [{device}]')

    model = models.load(model_name, shape_in=input_shape, shape_out=len(dataset.classes))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    if checkpoint is not None:
        checkpoint = torch.load(f'{data_path}/models/{model_name}/{checkpoint}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    experiment = Tracker('image_classification', experiment_config=params, data_path=data_path,experiment_path=experiment_path,dataset=dataset,model=model,optimizer=optimizer, use_wandb=True)
        

    for epoch in range(num_epochs):
        
        loss_train = train(dataset.train_loader, criterion, optimizer)

        accuracy_val, loss_val, labels_pred = evaluate(dataset.test_loader, model, criterion)

        experiment.log(epoch, scalars={'loss_val':loss_val,'loss_train':loss_train, 'accuracy_val':accuracy_val}, labels_pred=labels_pred)
