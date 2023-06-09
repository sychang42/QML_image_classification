import torchvision 
from torchvision.transforms import transforms

from datasets import *

def load_data(root, data, download = False, transform = None) : 
    
    if transform is None : 
        transform = transforms.Compose([transforms.ToTensor()])

    switcher = {
        "MNIST" : torchvision.datasets.MNIST,
        "FashionMNIST" : torchvision.datasets.FashionMNIST, 
        "SAT4" : SAT4, 
        "EuroSAT" : EuroSAT
    }

    dataset = switcher.get(data, lambda: None)
    
    trainds = dataset(root= root, train=True, download=download, transform=transform) 
    testds = dataset(root= root, train=False, download=download, transform=transform)

    return trainds, testds