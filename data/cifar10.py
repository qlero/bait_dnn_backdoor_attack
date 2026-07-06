"""
DATASET MODULE FOR LOADING THE CIFAR10 DATASET
"""

import torch
import torchvision
import torchvision.transforms as T

from torch.utils.data import DataLoader, Subset

###############################
# data augmentation pipelines #
###############################

cifar_mean = [0.5] * 3
cifar_std  = [0.5] * 3

normalizer   = T.Normalize(mean = cifar_mean, std = cifar_std)
unnormalizer = T.Lambda(lambda img: (img + 1) / 2)

base_transforms = T.Compose([
    T.Resize((64,64)),
    T.ToTensor()
])

train_transforms = T.Compose([
    T.RandomCrop(64, padding=4, fill=0.5),
    T.RandomHorizontalFlip(),
    T.RandomRotation((0, 15), fill=0.5),
    T.ColorJitter(brightness=0.3, hue=0.1),
    normalizer
])

test_transforms = T.Compose([
    normalizer
])

####################
# Loader functions #
####################

def setup_dataloaders(batch_size):
    """
    Loads (and in case downloads) the CIFAR10 datasets and creates
    both training and test dataloaders
    """
    print("Loading datasets......")
    # Sets torchvision datasets
    train_set = torchvision.datasets.CIFAR10(
        root = "./cifar10/", 
        train = True, download = True, 
        transform = base_transforms
    )
    test_set = torchvision.datasets.CIFAR10(
        root = "./cifar10/", 
        train = False, download = True, 
        transform = base_transforms
    )
    # Sets torchvision dataloaders
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_loader  = DataLoader(test_set, batch_size = 32, shuffle = False)
    print("Loading datasets DONE.")

    return train_loader, test_loader