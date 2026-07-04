"""
DATASET MODULE FOR LOADING THE CIFAR10 DATASET
"""

import torch
import torchvision
import torchvision.transforms as T

###############################
# data augmentation pipelines #
###############################

cifar_mean = [0.4914, 0.4822, 0.4465]
cifar_std  = [0.2023, 0.1994, 0.2010]

base_transforms = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])

train_transforms = T.Compose([
    T.RandomCrop(64, padding=4),
    T.RandomHorizontalFlip(),
    T.Normalize(mean = cifar_mean, std = cifar_std)
])

test_transforms = T.Compose([
    T.Normalize(mean = cifar_mean, std = cifar_std)
])

####################
# Loader functions #
####################

def setup_dataloaders():
    """
    Loads (and in case downloads) the CIFAR10 datasets and creates
    both training and test dataloaders
    """
    print("Loading datasets......")
    trainset = torchvision.datasets.CIFAR10(
        root = "./data/", 
        train = True, download = True, 
        transform = base_transforms
    )
    testset = torchvision.datasets.CIFAR10(
        root = "./data/", 
        train = False, download = True, 
        transform = base_transforms
    )
    print("Loading datasets DONE.")