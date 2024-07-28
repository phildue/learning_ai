
from typing import Dict
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
import numpy as np
import os
import random
class TrackerTest:
    def __init__(self, project_name:str, experiment_name:str, experiment_config:Dict, experiment_path, dataset, model, use_wandb=False):

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
        self.writer = SummaryWriter(experiment_path)
        self.loss_val = [np.inf]
        self.acc_val = [0]

        self.dataset = dataset
        self.model = model



    def log(self, step, scalars:Dict, labels_pred):
        scalars_str = ''
        for key, value in scalars.items():
            scalars_str += f'| {key}={value:.4f} '

        
        if self.use_wandb:
            for name, value in scalars.items():
                wandb.log({name:value},step=step)
             
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)
        
        acc_val = scalars['accuracy_val']

        if self.use_wandb:
            for i, ((images, labels_gt), labels_pred) in enumerate(zip(self.dataset.test_img_loader, labels_pred)):
                wandb.log({f'test_samples/img_{i:04d}': wandb.Image(images[0], caption=f'Gt={self.dataset.classes[labels_gt[0]]}, Pred={self.dataset.classes[labels_pred[0]]}'),
                            },step=step)
            self.run.summary['best_accuracy'] = acc_val
                    #torch.jit.script(model).save(f'{experiment_path}/model_best.ts')
                

        self.writer.flush()
        self.acc_val += [acc_val]
        