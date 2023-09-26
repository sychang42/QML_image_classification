"""
Function to load features that are extracted from the original images. 
"""
import os
import sys
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))


# Supported dimensionality reduction methods
_dim_red_type = ["pca", "ae", "deepae", "verydeepae"]

# Support dataset types
_datas = ["MNIST", "EuroSAT"]


def load_data(
    root: str,
    data: str,
    method: str,
    n_components: int = 16,
    classes: Optional[Union[List, np.ndarray]] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Loads features that have been extracted from original images using
    a dimensionality reduction method. The loaded features are used for quantum
    classifier training.

    Args:
        root (str): The root directory where the reduced dataset is located.
        method (str): The dimensionality reduction method employed for the dataset.
        n_components (int): The size of the loaded feature vectors.
        classes (Optional[Union[List, np.ndarray]]): The specific data classes to be
                         included for training. If set to None, all classes are used.

    Returns:
        X_train (np.ndarray): Training features of the shape (n_train, n_components).
        Y_train (np.ndarray): Training labels of the shape (n_train, ).
        X_test (np.ndarray): The test features of the shape (n_test, n_components).
        Y_test (np.ndarray): The test labels of the shape (n_test, ).
    """
    if classes is not None and type(classes) is not np.ndarray:
        classes = np.array(classes)

    assert data in _datas
    assert method in _dim_red_type

    file_name = data + "_" + str(n_components) + "components_" + method

    train_data = pd.read_csv(os.path.join(root, file_name + "_train.csv"))
    Y_train = train_data["label"].to_numpy()
    X_train = train_data.to_numpy()[:, :n_components]

    test_data = pd.read_csv(os.path.join(root, file_name + "_test.csv"))
    Y_test = test_data["label"].to_numpy()
    X_test = test_data.to_numpy()[:, :n_components]

    if classes is not None:
        mask_train = np.isin(Y_train, classes)
        mask_test = np.isin(Y_test, classes)

        X_train = X_train[mask_train]
        Y_train = Y_train[mask_train]
        X_test = X_test[mask_test]
        Y_test = Y_test[mask_test]

        for c_pos, c in enumerate(classes):
            Y_train[Y_train == c] = c_pos
            Y_test[Y_test == c] = c_pos

    return X_train, Y_train, X_test, Y_test
