import argparse
import json
import os
import sys
from math import ceil, log2

import numpy as np
import yaml
from dataset import load_data
from models.train import train

sys.path.append(os.path.dirname(__file__))


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # Retrieve argument from yaml file
    parser = argparse.ArgumentParser(description="Run qGAN experiments on simulator")
    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        default="configs/training.yaml",
    )

    args = parser.parse_args()

    with open(args.filename, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    data = config["training_params"]["data"]
    classes = config["training_params"]["classes"]  # Classes to train
    method = config["training_params"]["method"]  # Dimensionality reduction method
    n_components = config["training_params"]["n_components"]

    if "BCE_loss" in config["training_params"]["loss_type"]:
        config["model_params"]["num_measured"] = int(ceil(log2(len(classes))))
    else:
        config["model_params"]["num_measured"] = 1

    save_dir = config["logging_params"]["save_dir"]  # Directory to save
    seed = np.random.randint(10)
    config["training_params"]["seed"] = seed

    snapshot_dir = None
    if save_dir is not None:
        # Create folder to save training results
        path_count = 1
        snapshot_dir = os.path.join(save_dir, "run" + str(path_count))
        while os.path.exists(snapshot_dir):
            path_count += 1
            snapshot_dir = os.path.join(save_dir, "run" + str(path_count))

        os.makedirs(snapshot_dir)
        json.dump(config, open(os.path.join(snapshot_dir, "summary.json"), "+w"))

    # Load data
    load_dir = "/data/suchang/sy_phd/v4_QML_EO/v1_classification/data/"

    X_train, Y_train, X_test, Y_test = load_data(
        load_dir, data, method, n_components, classes
    )
    train_ds = {"image": X_train, "label": Y_train}
    test_ds = {"image": X_test, "label": Y_test}

    train(
        train_ds,
        test_ds,
        config["training_params"],
        config["model_params"],
        config["optim_params"],
        snapshot_dir,
    )
