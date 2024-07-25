import numpy as np
import cv2 as cv
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

while True:
    indices = torch.randperm(len(dataset))[:128]
    images = [dataset[i][0] for i in indices]
  
    image = torchvision.utils.make_grid(images,nrow=24)
   
    cv.imshow('CIFAR-10 Image Grid', image.numpy().transpose(1, 2, 0))

    if cv.waitKey(0) != 13:  # Enter key
        break

cv.destroyAllWindows()
