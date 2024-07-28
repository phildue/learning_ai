# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from ai.experiment_tracking.tracker import Tracker
from ai.datasets.cifar10 import Cifar10
from ai.models import models
SEED=666
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(10)


def train(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, criterion, optimizer:torch.optim.Optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i, (images, labels) in enumerate(tqdm(data_loader,desc='Training')):  
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss


def evaluate(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg : DictConfig) -> None:
    print(f'Running training with: {OmegaConf.to_yaml(cfg)}')

    datasets = {
        'cifar10': Cifar10
    }
    dataset = datasets['cifar10'](shape=(cfg.height, cfg.width), batch_size=cfg.batch_size, data_path=cfg.data_path)
    
    experiment_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Selected device: [{device}]')

    model = models.load(cfg.model_name, shape_in=(cfg.height, cfg.width), shape_out=len(dataset.classes))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)  

    if len(cfg.checkpoint) > 1:
        checkpoint = torch.load(cfg.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    experiment = Tracker('image_classification', 
                         experiment_config=OmegaConf.to_object(cfg),
                         experiment_name=cfg.experiment_name, 
                         data_path=cfg.data_path,
                         experiment_path=experiment_path,
                         dataset=dataset,
                         model=model,
                         optimizer=optimizer,
                         use_wandb=cfg.use_wandb)
        

    for epoch in range(cfg.num_epochs):
        
        loss_train = train(model, dataset.train_loader, criterion, optimizer)

        accuracy_val, loss_val, labels_pred = evaluate(model, dataset.test_loader, criterion)

        experiment.log(epoch, scalars={'loss_val':loss_val,'loss_train':loss_train, 'accuracy_val':accuracy_val}, labels_pred=labels_pred)


if __name__ == "__main__":
    main()
     

