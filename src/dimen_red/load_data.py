r"""
Function to load the given data and return PyTorch dataset.
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

import torch
import torchvision
from datasets import SAT4, EuroSAT
from torchvision.transforms import transforms
from typing import Optional, Callable, Tuple


def load_data(
    root: str,
    data: str,
    download: Optional[bool] = False,
    transform: Optional[Callable] = None,
) -> Tuple[torch.utils.data.Dataset]:
    r""" Load the specified dataset and create PyTorch datasets.

    Args:
        root (str): The root directory where the dataset is located.
            If the dataset does not exist locally, it will be downloaded here.
        data (str): The name of the dataset to load.
        download (bool): Boolean to indicate whether to download the dataset
            if it's not already available locally.
        transform (Optional[Callable]): A PyTorch transform function that takes a PIL image and 
            returns the transformed version.If None, it defaults to transform.ToTensor().

    Returns:
        Tuple[torch.utils.data.Dataset]: Tuple of the training and the the testing dataset.
    """

    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    switcher = {
        "MNIST": torchvision.datasets.MNIST,
        "FashionMNIST": torchvision.datasets.FashionMNIST,
        "SAT4": SAT4,
        "EuroSAT": EuroSAT,
    }

    dataset = switcher.get(data, lambda: None)

    if dataset is not None:
        trainds = dataset(root=root, train=True, download=download, transform=transform)
        testds = dataset(root=root, train=False, download=download, transform=transform)
    else:
        raise NotImplementedError("Dataset not supported.")

    return trainds, testds  # type: ignore
