import torch
from ai.cnn import ConvNeuralNet
import torchvision
import os
import torchvision.transforms as transforms
import random
import numpy as np
import cv2 as cv
# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20
data_path = './data/'
model_path = f'{data_path}/models/cnn'
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
# Create Testing dataset
test_dataset = torchvision.datasets.CIFAR10(root = './data',
                                            train = False,
                                            transform = all_transforms,
                                            download=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

model = torch.jit.load(f'{model_path}/model_best.ts')
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        idx = int(random.uniform(0, len(images-1)))
        img = images[idx].numpy().transpose(1, 2, 0)
        img = cv.copyMakeBorder(img, 0, 20, 0, 0, cv.BORDER_CONSTANT, value=[255, 255, 255])
        cv.putText(img, str(predicted[idx]), (img.shape[1] // 2 - 10, img.shape[0] - 5),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.imshow('Example', img)
        if cv.waitKey(0) != 13:  # Enter key
            break

    print(f'Accuracy of the network on the {len(test_loader)} test images: {100 * correct / total} %')


