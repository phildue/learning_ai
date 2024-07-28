import torch
import torchvision
import torchvision.transforms.v2 as transforms
import numpy as np

class Cifar10:
    def __init__(self,shape, batch_size, data_path):
        self.mean = np.array((0.4914, 0.4822, 0.4465))
        self.std = np.array((0.2023, 0.1994, 0.2010))
        self.classes = ['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.shape = shape

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(shape, padding=4),
            transforms.Resize(shape),
            transforms.RandomHorizontalFlip(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float64,scale=True),
            transforms.Normalize(self.mean, self.std),
        ])


        self.train_dataset = torchvision.datasets.CIFAR10(root = data_path,
                                                    train = True,
                                                    transform = self.transform_train,
                                                    download = True)

        self.train_loader = torch.utils.data.DataLoader(dataset = self.train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True)


        self.transform_test = transforms.Compose([
            transforms.Resize(shape),
            transforms.ToImage(),
            transforms.ToDtype(torch.float64,scale=True),
            transforms.Normalize(self.mean, self.std),
        ])

        self.test_dataset = torchvision.datasets.CIFAR10(root = data_path,
                                                    train = False,
                                                    transform = self.transform_test,
                                                    download=True)

        self.test_loader = torch.utils.data.DataLoader(dataset = self.test_dataset,
                                                batch_size = batch_size,
                                                shuffle = False)
        
        self.transform_test_img = transforms.Compose([
            transforms.Resize(shape),
            transforms.ToImage(),
            transforms.ToDtype(torch.float64,scale=True)
        ])
        
        self.test_img_dataset = torchvision.datasets.CIFAR10(root = data_path,
                                                    train = False,
                                                    transform = self.transform_test_img,
                                                    download=True)

        self.test_img_loader = torch.utils.data.DataLoader(dataset = self.test_img_dataset,
                                                batch_size = batch_size,
                                                shuffle = False)
        

