# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from ai.experiment_tracking.tracker_test import TrackerTest
from ai.datasets.cifar10 import Cifar10
from ai.models import models
SEED=666
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(10)


def evaluate(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        correct = 0
        total = 0
        labels_all = []
        for images, labels in tqdm(data_loader,desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            labels_all += [predicted]

    return correct/total, labels_all
        


@hydra.main(version_base=None, config_path="../config", config_name="test")
def main(cfg : DictConfig) -> None:
    print(f'Running evaluation with: {OmegaConf.to_yaml(cfg)}')

    datasets = {
        'cifar10': Cifar10
    }
    dataset = datasets['cifar10'](shape=(cfg.height, cfg.width), batch_size=cfg.batch_size, data_path=cfg.data_path)
    
    experiment_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Selected device: [{device}]')

    model = models.load(cfg.model_name, shape_in=(cfg.height, cfg.width), shape_out=len(dataset.classes))
    model.eval()


    if len(cfg.checkpoint) > 1:
        checkpoint = torch.load(cfg.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

    experiment = TrackerTest('image_classification', 
                         experiment_config=OmegaConf.to_object(cfg),
                         experiment_name=f'{cfg.experiment_prefix}_{cfg.model_name}_{cfg.dataset_name}', 
                         experiment_path=experiment_path,
                         dataset=dataset,
                         model=model,
                         use_wandb=cfg.use_wandb)
        

    
    accuracy_val, labels_pred = evaluate(model, dataset.test_loader)

    experiment.log(step=0, scalars={'accuracy_val':accuracy_val}, labels_pred=labels_pred)


if __name__ == "__main__":
    main()
     

