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
    r"""pre-processed features obtained through dimensionality reduction from
    original images. The loaded features are used for training of the quantum
    classifier.

    Args:
        root (str): The root directory containing the reduced.
        method (str): The dimensionality reduction method used for the dataset.
        n_components (int): The dimensionality of the loaded feature vectors.
        classes (Union[List, np.ndarray], optional): Specific data classes to be included
            for training. If `None`, all available classes are used. `None` by default.

    Returns:
        Tuple[np.ndarray, ...]: Tuple of ``np.ndarray`` containing the training
        data/labels, and the test data/labels. The training/test data are vectors of
        size ``n_components``.
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
