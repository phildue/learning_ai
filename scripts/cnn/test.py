import torch
from ai.cnn import ConvNeuralNet
import torchvision
import os
import torchvision.transforms as transforms
import random
import numpy as np
import cv2 as cv
from tqdm import tqdm
# Define relevant variables for the ML task
batch_size = 64
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_classes = len(classes)
experiment_name = 'epoch-20-lr-0.001000-bs-64'
data_path = './data/'
model_path = f'{data_path}/models/cnn/{experiment_name}'
# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Selected device: [{device}]')

model = torch.jit.load(f'{model_path}/model_best.ts')
model.eval()

# Use transforms.compose method to reformat images for modeling,
# and save to variable all_transforms for later use
mean=[0.4914, 0.4822, 0.4465]
std=[0.2023, 0.1994, 0.2010]
all_transforms = transforms.Compose([transforms.Resize((32,32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=mean,
                                                          std=std)
                                     ])
#TODO: replace mean and std with live computation
# Create Testing dataset
test_dataset = torchvision.datasets.CIFAR10(root = data_path,
                                            train = False,
                                            transform = all_transforms,
                                            download=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)


with torch.no_grad():
    correct = 0
    total = 0
    examples = []
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        idx = int(random.uniform(0, len(images-1)))
        img = images[idx].numpy()
        img = img.transpose(1, 2, 0)
        img *= std
        img += mean
        img = cv.copyMakeBorder(img, 0, 20, 0, 0, cv.BORDER_CONSTANT, value=[255, 255, 255])
        cv.putText(img, classes[predicted[idx]], (img.shape[1] // 2 - 10, img.shape[0] - 5),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv.LINE_AA)
        examples += [torch.tensor(img.transpose(2, 0, 1))]
    
    examples = torchvision.utils.make_grid(examples,nrow=24).numpy().transpose(1, 2, 0)
    print(f'Accuracy of the network on the {len(test_loader)} test images: {100 * correct / total} %')
    cv.imshow('Example', (examples))
    cv.waitKey(0)


