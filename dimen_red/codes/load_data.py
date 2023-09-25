"""
Function to load the given data and return PyTorch dataset.
"""
import torch
import torchvision
from datasets import SAT4, EuroSAT
from torchvision.transforms import transforms
from typing import Optional, Callable, Tuple

import os
import sys
sys.path.append(os.path.dirname(__file__))


def load_data(root: str,
              data: str,
              download: Optional[bool] = False,
              transform: Optional[Callable] =None 
              ) -> Tuple[torch.utils.data.Dataset]:
    """
    Load the given data and return PyTorch dataset.

    Args : 
        root (str) : Root directory where the image data is located. The data will be downloaded in 
                    the directory if it does not exists.  
        data (str) : Name of the dataset to be loaded.
        download (bool) : Boolean to indicate whether we download the data or not.
        transform (Optional[Callable]) : PyTorch transform function which takes the PIL image and return
                                        the transformed version. Set to be transform.ToTensor() if None.

    Return : 
        trainds (torch.utils.data.Dataset) : Train dataset. 
        testds (torch.utils.data.Dataset]) : Test dataset.
    """

    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    switcher = {
        "MNIST": torchvision.datasets.MNIST,
        "FashionMNIST": torchvision.datasets.FashionMNIST,
        "SAT4": SAT4,
        "EuroSAT": EuroSAT
    }

    dataset = switcher.get(data, lambda: None)

    if dataset is not None : 
        trainds = dataset(root=root, train=True,
                        download=download, transform=transform)
        testds = dataset(root=root, train=False,
                        download=download, transform=transform)
    else : 
        raise NotImplementedError("Dataset not supported.")
    
    return trainds, testds
