"""
Utility functions used for quantum classifier training. 
"""

import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from jax import Array


def save_outputs(epoch: int, snapshot_dir: str, outputs: Array, labels: Array) -> None:
    """
    Saves the classifier's predictions and the corresponding ground truth labels at
    a specific training epoch as a CSV file.

    Args:
        epoch (int): The current training epoch.
        snapshot_dir (str): The directory where the results will be saved.
        outputs (jax.Array): The predictions made by the classifier at the
                             specified epoch.
        labels (jax.Array): The ground truth labels.

    Returns:
        None
    """

    df = pd.DataFrame({"preds": outputs})
    df["labels"] = labels

    df.to_csv(os.path.join(snapshot_dir, "classification_epoch" + str(epoch) + ".csv"))


def print_losses(
    epoch: int, epochs: int, train_loss: Dict[str, Array], valid_loss: Dict[str, Array]
) -> None:
    """
    Print the training and validation losses.

    Args :
        epoch (int) : Current epoch.
        epochs (int) : Total number of epohcs.
        train_loss (dict) : Training loss.
        valid_loss (dict) : Validation loss.
    Return :
        None
    """
    print(
        f"Epoch : {epoch + 1}/{epochs}, Train loss (average) = "
        f", ".join("{}: {}".format(k, v) for k, v in train_loss.items())
    )
    print(
        f"Epoch : {epoch + 1}/{epochs}, Valid loss = "
        f", ".join("{}: {}".format(k, v) for k, v in valid_loss.items())
    )


def plot_result(train_loss, test_loss, snapshot_dir):
    N = len(train_loss.keys())
    fig, axes = plt.subplots(1, N, figsize=(N * 5, 4))

    for ax, k in zip(axes, train_loss.keys()):
        epoch = range(len(train_loss[k]))
        plt.plot(epoch, train_loss[k], label="Train")
        plt.plot(epoch, test_loss[k], label="Test")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(k)

        ax.legend()

    plt.savefig(fig, os.path.join(snapshot_dir, "loss.png"))
