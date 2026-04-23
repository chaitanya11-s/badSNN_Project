import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from spikingjelly.datasets.n_mnist import NMNIST

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def get_cifar10_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root=Config.DATA_ROOT, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=Config.DATA_ROOT, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_cifar100_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = datasets.CIFAR100(root=Config.DATA_ROOT, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root=Config.DATA_ROOT, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_gtsrb_loaders():
    # GTSRB images are of varying sizes, commonly resized to 32x32 for SNN evaluations
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)),
    ])

    trainset = datasets.GTSRB(root=Config.DATA_ROOT, split='train', download=True, transform=transform_train)
    testset = datasets.GTSRB(root=Config.DATA_ROOT, split='test', download=True, transform=transform_test)
    
    train_loader = DataLoader(trainset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_nmnist_loaders():
    # SpikingJelly N-MNIST produces tensors of shape [T, C, H, W] if data_type='frame'
    nmnist_root = os.path.join(Config.DATA_ROOT, 'nmnist')
    
    trainset = NMNIST(root=nmnist_root, train=True, data_type='frame', 
                      frames_number=Config.TIMESTEPS, split_by='number')
    testset = NMNIST(root=nmnist_root, train=False, data_type='frame', 
                     frames_number=Config.TIMESTEPS, split_by='number')

    train_loader = DataLoader(trainset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_dataloaders():
    if Config.DATASET == 'cifar10':
        return get_cifar10_loaders()
    elif Config.DATASET == 'cifar100':
        return get_cifar100_loaders()
    elif Config.DATASET == 'gtsrb':
        return get_gtsrb_loaders()
    elif Config.DATASET == 'nmnist':
        return get_nmnist_loaders()
    else:
        raise ValueError(f"Unknown dataset: {Config.DATASET}")
