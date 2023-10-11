import argparse
import json
import os
import sys
from math import ceil, log2

sys.path.insert(0, os.path.abspath("../src/"))

import numpy as np
import yaml

from dimen_red.dimensionality_reduction import dimensionality_reduction
from quantum_classifier.models.train import train

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
        default="configs/config.yaml",
    )

    parser.add_argument("--gpu", "-g", dest="gpu_num", help="GPU number", default=2)

    args = parser.parse_args()
    gpu = args.gpu_num

    with open(args.filename, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Configuration of classification
    save_dir = config["save_dir"]  # Directory to save
    seed = np.random.randint(10)
    config["quantum_classifier"]["training_params"]["seed"] = seed

    snapshot_dir = None
    if save_dir is not None:
        count = 1
        while os.path.exists(os.path.join(save_dir, "run" + str(count))):
            count += 1
        snapshot_dir = os.path.join(save_dir, "run" + str(count))
        os.makedirs(snapshot_dir)

        json.dump(config, open(os.path.join(snapshot_dir, "config.json"), "+w"))

    # Configuration for dimensionality reduction
    method = config["method"]
    data = config["data"]
    load_dir = config["load_dir"]
    n_components = config["num_components"]

    if method == "pca":
        hp = {"nz": n_components}
    elif method == "ae":
        hp = config["dimensionality_reduction"]
        hp["model_params"] = {"nz": n_components}
    else:
        raise NotImplementedError("Dimensionlity reduction method not implemented")

    # Configuration for quantum classifier
    config = config["quantum_classifier"]
    classes = config["training_params"]["classes"]  # Classes to train

    if "BCE_loss" in config["training_params"]["loss_type"]:
        config["model_params"]["num_measured"] = int(ceil(log2(len(classes))))
    else:
        config["model_params"]["num_measured"] = 1

    # Load data

    print(f"Start dimensionlity reduction on {data} dataset with {method} method")

    train_loss, test_loss, X_train, Y_train, X_test, Y_test = dimensionality_reduction(
        load_dir, data, method, hp, gpu=gpu, snapshot_dir=snapshot_dir
    )

    print(
        f"Dimensionality reduction finished with Train loss = {train_loss:.2e}, "
        f"Valid loss = {test_loss:.2e}"
    )

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

    print(f"Start feature classification")

    train_ds = {"image": X_train, "label": Y_train}
    test_ds = {"image": X_test, "label": Y_test}
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    train(
        train_ds,
        test_ds,
        config["training_params"],
        config["model_params"],
        config["optim_params"],
        snapshot_dir,
    )
