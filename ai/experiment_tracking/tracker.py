
from typing import Dict
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
import numpy as np
import os
import random
class Tracker:
    def __init__(self, project_name:str, experiment_name:str, experiment_config:Dict, experiment_path, data_path, dataset, model, optimizer, use_wandb=False):

        if use_wandb:
            self.run = wandb.init(
                project=project_name,
                name=experiment_name,
                config=experiment_config
            )
        self.experiment_config = experiment_config
        self.model_name = experiment_config['model_name']
        self.dataset_name = experiment_config['dataset_name']
        self.experiment_name = experiment_name
        self.experiment_path = experiment_path       
        self.use_wandb = use_wandb
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(f'{experiment_path}', exist_ok=True)
        self.writer = SummaryWriter(experiment_path)
        self.loss_val = [np.inf]
        self.acc_val = [0]

        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        if self.use_wandb:
            wandb.watch(self.model)



    def log(self, epoch, scalars:Dict, labels_pred):
        scalars_str = ''
        for key, value in scalars.items():
            scalars_str += f'| {key}={value:.4f} '

        print(f'Epoch [{epoch+1}]: {scalars_str}')
        
        if self.use_wandb:
            wandb.log({'epoch':epoch},step=epoch)
            for name, value in scalars.items():
                wandb.log({name:value},step=epoch)
             
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, epoch)
        
        if epoch % 10 == 0:
            filepath = f'{self.experiment_path}/checkpoint_{epoch:04d}.pt'
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scalars': scalars,
                    }, filepath)
            if self.use_wandb:
                wandb.log_artifact(filepath, name=f'{self.model_name}-{self.dataset_name}-{self.experiment_name}-checkpoint-{epoch:04d}', type='model') 


        loss_val = scalars['loss_val']
        acc_val = scalars['accuracy_val']

        acc_val_max = max(self.acc_val)
        if epoch == 0 or acc_val > acc_val_max:
                print(f'Saving new best model at epoch [{epoch}] with accuracy [{acc_val}] > [{acc_val_max}]')
                filepath = f'{self.experiment_path}/checkpoint_best.pt'
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scalars': scalars,
                        }, filepath)
                if self.use_wandb:
                    wandb.log_artifact(filepath, name=f'{self.model_name}-{self.dataset_name}-{self.experiment_name}-best', type='model')
                    for i, ((images, labels_gt), labels_pred) in enumerate(zip(self.dataset.test_img_loader, labels_pred)):
                        wandb.log({f'test_samples/img_{i:04d}': wandb.Image(images[0], caption=f'Gt={self.dataset.classes[labels_gt[0]]}, Pred={self.dataset.classes[labels_pred[0]]}'),
                                   },step=epoch)
                    self.run.summary['best_accuracy'] = acc_val
                #torch.jit.script(model).save(f'{experiment_path}/model_best.ts')
            

        self.writer.flush()
        self.loss_val += [loss_val]
        self.acc_val += [acc_val]
        