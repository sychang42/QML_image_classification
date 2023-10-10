"""
SAT4 and EuroSAT dataset
The code follows the same code as the one given in PyTorch for MNIST
dataset [1].

[1]: https://pytorch.org/vision/0.15/generated/torchvision.datasets.MNIST.html 
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))


import pickle
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.error import URLError

import imageio
import numpy as np
import torch
from PIL import Image
from skimage import img_as_ubyte
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


class SAT4(VisionDataset):
    """`SAT4 <https://www.kaggle.com/datasets/crawford/deepsat-sat4>`_ Dataset.

    Args:
        root (str): Root directory of dataset where ``X_train.pkl`` and ``X_test.pkl`` exist.
        train (bool, optional): If True, creates dataset from ``X_train.pkl`,
            otherwise from ``X_test.pkl``.
        download (bool, optional): If True, downloads the dataset from the designated
            url to the root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (Callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (Callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    # List[str]: List of SAT4 class names
    classes = ["Barren Land", "Trees", "Grassland", "Others"]

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

        self.raw_folder = os.path.join(
            self.root, "SAT4"
        )  # Folder where the dataset exists
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets = self._load_data()

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load data into torch.Tensor from numpy.ndarray

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of image data and image labels.
        """
        data, targets = self._load_data_numpy()
        data = torch.Tensor(data)
        targets = torch.Tensor(targets)

        return data, targets

    def _load_data_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from the original ``pickle`` file, into ``np.ndarray``.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of image data and image labels. If
            ``self.train`` is ``True``, the function returns the training set; otherwise,
            it returns the test set.
        """

        image_file = f"{'X_train' if self.train else 'X_test'}_sat4.pkl"
        data = pickle.load(
            open(os.path.join(self.raw_folder, image_file), "rb")
        ).to_numpy()
        data = data.reshape((-1, 28, 28, 4)).astype(float)
        data = np.transpose(np.array(data / 255.0), (0, 3, 1, 2))

        label_file = f"{'Y_train' if self.train else 'Y_test'}_sat4.pkl"
        targets = pickle.load(
            open(os.path.join(self.raw_folder, label_file), "rb")
        ).to_numpy()
        targets = np.array(np.argmax(targets, axis=1))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index.

        Returns:
            Tuple[Any, Any]: (image, target) at the index number.
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
        """
        Download the SAT4 data if it doesn't exist in the root directory.
        """

        if self._check_exists():
            print("Files already downloaded and verified")
            return

        print(
            "Download the dataset from https://www.kaggle.com/datasets/crawford/deepsat-sat4"
        )

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class EuroSAT(VisionDataset):
    r"""`EuroSAT <https://github.com/phelber/EuroSAT>`_ Dataset.

    Args:
        root (str): Root directory of dataset where the image folders exist.
        train (bool, optional): If True, creates train dataset, otherwise test dataset.
            Defaults to True.
        download (bool, optional): If True, downloads the dataset from the designated
            url to the root directory. If dataset is already downloaded, it is not
            downloaded again. Defaults to False.
        transform (Callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``. Defaults
            to None.
        target_transform (Callable, optional): A function/transform that takes in the
            target and transforms it. Defaults to None.
    """

    # str: URL of the EuroSAT dataset
    _download_url = "https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip?download=1"

    # Tuple[str] : File name and md5 value for download
    _resources = ("EuroSAT_RGB.zip", "f46e308c4d50d4bf32fedad2d3d62f3b")

    # List[str]: List of EuroSAT class names
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
        "SeaLake",
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

        self.train = train
        self.num_images = 27000
        self.img_size = [64, 64]
        self.num_channel = 3

        self.seed = 42
        self.test_ratio = 0.2

        self.raw_folder = os.path.join(
            self.root, "EuroSAT"
        )  # Folder where the dataset exists

        if download:
            self.download()

        self.data, self.targets = self._load_data()

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return os.path.exists(self.raw_folder)

    def download(self) -> None:
        r"""Download the EuroSAT data if it doesn't exist already."""

        if self._check_exists():
            return

        # download files
        filename = self._resources[0]
        md5 = self._resources[1]

        try:
            print(f"Downloading {self._download_url}")
            download_and_extract_archive(
                self._download_url, download_root=self.root, filename=filename, md5=md5
            )
            os.rename(os.path.join(self.root, "EuroSAT_RGB"), self.raw_folder)

        except URLError as error:
            print(f"Failed to download (trying next):\n{error}")

    def _load_data(self) -> Tuple[Any, Any]:
        r"""Load data from the specified root directory, splitting it into test and train
        sets based on a seed. Images are resized by default to (64, 64).

        Returns:
            Tuple[Any, Any]: Tuple of image data and image labels. If ``self.train`` is
            ``True``, the function returns the training set; otherwise, it returns the
            test set.
        """

        images = np.zeros(
            [self.num_images, self.img_size[0], self.img_size[1], self.num_channel],
            dtype="uint8",
        )
        labels = []

        idx = 0
        # read all the files from the image folder
        for label, f in enumerate(os.listdir(self.raw_folder)):
            path = os.path.join(self.raw_folder, f)
            if os.path.isfile(path):
                continue
            for f2 in os.listdir(path):
                sub_path = os.path.join(path, f2)
                # Resize images with pixel off
                image = imageio.v3.imread(sub_path)
                if (
                    image.shape[0] != self.img_size[0]
                    or image.shape[1] != self.img_size[1]
                ):
                    # print("Resizing image...")
                    image = img_as_ubyte(
                        resize(
                            image,
                            (self.img_size[0], self.img_size[1]),
                            anti_aliasing=True,
                        )
                    )

                images[idx] = img_as_ubyte(image)
                labels.append(label)

                idx = idx + 1

        labels = np.asarray(labels)

        # Split the data into train & testset
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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index.

        Returns:
            Tuple[Any, Any]: (image, target) at the index number.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]
