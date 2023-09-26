"""
Function to perform dimensionality reduction method. 
Currently, only PCA and convolutional autoencoder are supported.
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from ae_vanilla import vanilla_autoencoder
from datasets import load_data
from sklearn.decomposition import PCA



def dimensionality_reduction(
    root: str,
    data: str,
    method: str,
    hp: Dict[str, Any],
    gpu: int = 0,
    snapshot_dir: str = None,
) -> Tuple[float]:
    """
    Perform dimensionality reduction on image data.

    Args:
        root (str): Root directory containing the image dataset.
        data (str): Input dataset to apply dimensionality reduction to.
        method (str): Method for dimensionality reduction. Currently, only PCA and convolutional autoencoder are supported.
        hp (Dict[str, Any]): Hyperparameters specific to the chosen dimensionality reduction method.
        gpu (int): ID of the GPU to use for autoencoder training. Set to None to run on CPU.
        snapshot_dir (str): Directory to store the output data. If None, results will not be saved.

    Returns:
        train_mse (float): Mean Squared Error (MSE) loss between the original and reconstructed images for the training set.
        test_mse (float): Mean Squared Error (MSE) loss between the original and reconstructed images for the test set.
    """

    # Load data.
    trainds, testds = load_data(root, data)

    img_shape = list(trainds.data.shape[1:])

    if method == "pca":
        X_train = trainds.data.reshape((len(trainds), -1))
        X_train = X_train.numpy() if type(X_train) == torch.Tensor else X_train

        # If the images are scaled between 0 and 255, rescale them between 0 and 1.
        if np.max(X_train) > 1.0:
            X_train = X_train / 255.0
        Y_train = (
            trainds.targets.numpy()
            if type(trainds.targets) == torch.Tensor
            else trainds.targets
        )

        X_test = testds.data.reshape((len(testds), -1))
        X_test = X_test.numpy() if type(X_test) == torch.Tensor else X_test
        if np.max(X_test) > 1.0:
            X_test = X_test / 255.0
        Y_test = (
            testds.targets.numpy()
            if type(testds.targets) == torch.Tensor
            else testds.targets
        )

        pca = PCA(n_components=hp["nz"])

        pca = pca.fit(X_train)

        if snapshot_dir is not None:
            with open(os.path.join(snapshot_dir, "pca.pkl"), "wb") as file:
                pickle.dump(pca, file)

        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Rescale the pca output between 0 and 1.
        X_train_pca_rescaled = (X_train_pca - X_train_pca.min()) / (
            X_train_pca.max() - X_train_pca.min()
        )
        X_test_pca_rescaled = (X_test_pca - X_test_pca.min()) / (
            X_test_pca.max() - X_test_pca.min()
        )

        recons_train = pca.inverse_transform(X_train_pca)
        recons_test = pca.inverse_transform(X_test_pca)

        train_mse = np.mean((recons_train - X_train) ** 2)
        test_mse = np.mean((recons_test - X_test) ** 2)

        img_shape.insert(0, -1)
        pred_train = (X_train_pca_rescaled, recons_train.reshape(img_shape), Y_train)
        pred_test = (X_test_pca_rescaled, recons_test.reshape(img_shape), Y_test)

    elif method == "autoencoder":
        if len(img_shape) == 2:
            img_shape = [1, img_shape[0], img_shape[1]]

        if img_shape[2] < img_shape[0]:
            img_shape = [img_shape[2], img_shape[0], img_shape[1]]

        hp["model_params"]["img_shape"] = img_shape
        num_epoch = hp["training_params"]["num_epoch"]
        batch_size = hp["training_params"]["batch_size"]

        # Load the data into PyTorch dataloader.
        trainloader = torch.utils.data.DataLoader(
            trainds, batch_size=batch_size, shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            testds, batch_size=batch_size, shuffle=True
        )

        # Device to run the training.
        device = torch.device(
            "cuda:" + str(gpu) if torch.cuda.is_available() else "cpu"
        )
        model = vanilla_autoencoder(device, hp, snapshot_dir)
        model.train_model(num_epoch, trainloader, testloader)

        pred_train = model.predict(trainloader)
        pred_test = model.predict(testloader)

        train_mse = model.train_loss[np.argmin(model.train_loss)]
        test_mse = model.valid_loss[np.argmin(model.valid_loss)]

    else:
        raise NotImplementedError("Method Not implemented")

    # Restore the results in snapshot directory
    if snapshot_dir is not None:
        df = pd.DataFrame(pred_train[0])
        df["label"] = pred_train[2]
        df.to_csv(os.path.join(snapshot_dir, "train_features.csv"))

        df = pd.DataFrame(pred_test[0])
        df["label"] = pred_test[2]
        df.to_csv(os.path.join(snapshot_dir, "test_features.csv"))

    return train_mse, test_mse
