"""
Utility functions used for quantum classifier training. 
"""

import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from jax import Array


def save_outputs(epoch: int, snapshot_dir: str, outputs: Array, labels: Array) -> None:
    r"""Saves the classifier's predictions and the corresponding ground truth labels at
    a specific training epoch as a CSV file.

    Args:
        epoch (int): The current training epoch.
        snapshot_dir (str): The directory where the results will be saved.
        outputs (jax.Array): The predictions made by the classifier at the
                             specified epoch.
        labels (jax.Array): The ground truth labels.
    """

    df = pd.DataFrame({"preds": outputs})
    df["labels"] = labels

    df.to_csv(os.path.join(snapshot_dir, "classification_epoch" + str(epoch) + ".csv"))


def print_losses(
    epoch: int, epochs: int, train_loss: Dict[str, float], valid_loss: Dict[str, float]
) -> None:
    r"""Print the training and validation losses.

    Args:
        epoch (int) : Current epoch.
        epochs (int) : Total number of epohcs.
        train_loss (dict) : Training loss.
        valid_loss (dict) : Validation loss.
    """
    print(
        f"Epoch : {epoch + 1}/{epochs}, Train loss (average) = "
        f", ".join("{}: {}".format(k, v) for k, v in train_loss.items())
    )
    print(
        f"Epoch : {epoch + 1}/{epochs}, Valid loss = "
        f", ".join("{}: {}".format(k, v) for k, v in valid_loss.items())
    )
