import os
import os.path
import sys
sys.path.append(os.path.dirname(__file__)) 


import shutil
import string
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError

import imageio
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_float32

import numpy as np
import torch
from PIL import Image

import pickle

import torchvision 
from torchvision.transforms import transforms

from torchvision.datasets.vision import VisionDataset   
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, extract_archive, verify_str_arg


class SAT4(VisionDataset):
    """`SAT4 <https://www.kaggle.com/datasets/crawford/deepsat-sat4>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
            and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes = [
        "Barren Land",
        "Trees",
        "Grassland",
        "Others"
    ]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set
        
        self.raw_folder = os.path.join(self.root, "SAT4")
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()


    def _load_data(self):
        
        data, targets = self._load_data_numpy() 
        data = torch.Tensor(data) 
        targets = torch.Tensor(targets) 
        
        return data, targets
    
    def _load_data_numpy(self):
        image_file = f"{'X_train' if self.train else 'X_test'}_sat4.pkl"
        data = pickle.load(open(os.path.join(self.raw_folder, image_file), 'rb')).to_numpy() 
        data = data.reshape((-1, 28, 28, 4)).astype(float)
        data = np.transpose(np.array(data/255.), (0,3,1,2))
           
        label_file = f"{'Y_train' if self.train else 'Y_test'}_sat4.pkl"
        targets = pickle.load(open(os.path.join(self.raw_folder, label_file), 'rb')).to_numpy() 
        targets = np.array(np.argmax(targets, axis = 1))  
        
        return data, targets
    
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        
        return img, target


    def __len__(self) -> int:
        return len(self.data)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}   
    
    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.root, "SAT4"))

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            print("Files already downloaded and verified")
            return

        print("Download the dataset from https://www.kaggle.com/datasets/crawford/deepsat-sat4")

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
    
    
class EuroSAT(VisionDataset):
            
    download_url = "https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip?download=1"
    
    resources = ("EuroSAT_RGB.zip", "f46e308c4d50d4bf32fedad2d3d62f3b")
    

    classes = [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake"
    ]        
        

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.root = root
        
        self.train = train
        self.num_images = 27000 
        self.img_size = [64, 64] 
        self.num_channel = 3
        self.transform = transform
        self.target_transform = target_transform
        self.seed = 42 
        self.test_ratio = 0.2
        self.img_folder = os.path.join(self.root, "EuroSAT_RGB")
        
        if download:
            self.download()
        
        
        self.data, self.targets = self._load_data()
    
    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return os.path.exists(self.img_folder)


    def download(self) -> None:
        """Download the EuroSAT data if it doesn't exist already."""

        if self._check_exists():
            return
            
        # download files
        filename = self.resources[0]
        md5 = self.resources[1]
        
        try:
            print(f"Downloading {self.download_url}")
            download_and_extract_archive(self.download_url, download_root=self.root, filename=filename, md5=md5)
        except URLError as error:
            print(f"Failed to download (trying next):\n{error}")

    
    def _load_data(self):
        """Loads the data from the passed root directory. Splits in test/train based on seed. By default resized to 256,256
        """
        images = np.zeros([self.num_images, self.img_size[0], self.img_size[1], 3], dtype="uint8")
        labels = []

        idx = 0 
        # read all the files from the image folder
        for label, f in enumerate(os.listdir(self.img_folder)):
            path = os.path.join(self.img_folder, f)
            if os.path.isfile(path):
                continue
            for f2 in os.listdir(path):
                sub_path = os.path.join(path, f2)
                # Resize images with pixel off 
                image = imageio.v3.imread(sub_path)
                if image.shape[0] != self.img_size[0] or image.shape[1] != self.img_size[1]:
                    # print("Resizing image...")
                    image = img_as_ubyte(
                        resize(image, (self.img_size[0], self.img_size[1]), anti_aliasing=True)
                    )
                
                images[idx] = img_as_ubyte(image)
                labels.append(label)
                
                idx = idx + 1

        labels = np.asarray(labels)
        
        # split into a train and test set as provided data is not presplit
        X_train, X_test, Y_train, Y_test = train_test_split(
            images,
            labels,
            test_size=self.test_ratio,
            random_state=self.seed,
            stratify=labels,
        )
        if self.train:
            return X_train, Y_train
        else:
            return X_test, Y_test
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]
    
    
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


