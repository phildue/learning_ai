import numpy as np
import cv2 as cv
import torch
import torchvision
import torchvision.transforms as transforms
data_path = './data/'
model_path = f'{data_path}/models/cnn'
transform = transforms.Compose([transforms.ToTensor()])
cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

def get_random_images(dataset, num_images):
    indices = torch.randperm(len(dataset))[:num_images]
    images = [dataset[i][0].numpy().transpose(1, 2, 0) for i in indices]
    return images

def create_image_grid(images, grid_size):
    rows = []
    for i in range(grid_size):
        row_images = images[i * grid_size:(i + 1) * grid_size]
        rows.append(np.hstack(row_images))
    grid_image = np.vstack(rows)
    return grid_image

while True:
    images = get_random_images(cifar10, 256)
    grid_image = create_image_grid(images, 8)
    resized_image = cv.resize(grid_image, (1280, 960))

    cv.imshow('CIFAR-10 Image Grid', resized_image)

    if cv.waitKey(0) != 13:  # Enter key
        break

cv.destroyAllWindows()
