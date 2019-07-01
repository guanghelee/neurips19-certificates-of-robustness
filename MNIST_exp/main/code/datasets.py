from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"

# list of all datasets
DATASETS = ["imagenet", "cifar10", "mnist"]


def get_dataset(dataset: str, split: str, batch_size = -1, num_workers = 2) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split, batch_size, num_workers=num_workers)
    elif dataset == "cifar10":
        return _cifar10(split, batch_size, num_workers=num_workers)
    elif dataset == 'mnist':
        return _mnist(split, batch_size, num_workers=num_workers)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "mnist":
        return 10


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == 'mnist':
        return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_MNIST_MEAN = [0.5]
_MNIST_STDDEV = [0.5]

def _cifar10(split: str, batch_size, num_workers=-1, pin_memory=False):
    print('num_workers =', num_workers)
    if split == "train":
        train_dataset = datasets.CIFAR10("../dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))

        test_dataset = datasets.CIFAR10("../dataset_cache", train=False, download=True, transform=transforms.ToTensor())

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=pin_memory)   
        return train_loader, test_loader
    elif split == "test":
        return datasets.CIFAR10("../dataset_cache", train=False, download=True, transform=transforms.ToTensor())


def _imagenet(split: str, batch_size, num_workers=-1, pin_memory=True):
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    print('num_workers =', num_workers)
    if split == "train":
        train_subdir = os.path.join(dir, "train")
        train_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        train_dataset = datasets.ImageFolder(train_subdir, train_transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory)

        test_subdir = os.path.join(dir, "val")
        test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])    
        test_dataset = datasets.ImageFolder(test_subdir, test_transform)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=pin_memory)    
        return train_loader, test_loader

    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        return datasets.ImageFolder(subdir, transform)

def _mnist(split: str, batch_size, random_seed=0, num_workers=-1, pin_memory=False):
    print('num_workers =', num_workers)
    if split == "train":
        train_dataset = datasets.MNIST(root='../dataset_cache', train=True, download=True, transform=transforms.ToTensor())
        valid_dataset = datasets.MNIST(root='../dataset_cache', train=True, download=True, transform=transforms.ToTensor())

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = 5000

        np.random.seed(random_seed)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        print('train.size={}'.format(len(train_idx)))
        print('valid.size={}'.format(len(valid_idx)))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        return train_loader, valid_loader
        # dataset = datasets.MNIST('../dataset_cache', train=True, download=True, transform=transforms.ToTensor())

    elif split == 'eval_train':
        total_dataset = datasets.MNIST(root='../dataset_cache', train=True, download=True, transform=transforms.ToTensor())

        num_train = len(total_dataset)
        indices = list(range(num_train))
        split = 5000

        np.random.seed(random_seed)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        
        dataset = []
        for idx in train_idx:
            dataset.append(total_dataset[idx])
        
        print('train.size={}'.format(len(dataset)))
        return dataset

    elif split == 'valid':
        total_dataset = datasets.MNIST(root='../dataset_cache', train=True, download=True, transform=transforms.ToTensor())

        num_train = len(total_dataset)
        indices = list(range(num_train))
        split = 5000

        np.random.seed(random_seed)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        
        dataset = []
        for idx in valid_idx:
            dataset.append(total_dataset[idx])
        
        print('valid.size={}'.format(len(dataset)))
        return dataset

    elif split == "test":
        dataset = datasets.MNIST('../dataset_cache', train=False, download=True, transform=transforms.ToTensor())
        
        print('test.size={}'.format(len(dataset)))
        return dataset



class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
