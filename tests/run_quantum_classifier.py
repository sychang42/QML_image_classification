import argparse
import json
import os
import sys
from math import ceil, log2

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath("../src/"))


from quantum_classifier.dataset import load_data
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

    parser.add_argument("--gpu", "-g", dest="gpu_num", help="GPU number", default="2")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    with open(args.filename, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    data = config["data"]
    method = config["method"]
    load_dir = config["load_dir"]

    n_components = config["num_components"]
    save_dir = config["save_dir"]  # Directory to save

    config = config["quantum_classifier"]
    classes = config["training_params"]["classes"]  # Classes to train

    if "BCE_loss" in config["training_params"]["loss_type"]:
        config["model_params"]["num_measured"] = int(ceil(log2(len(classes))))
    else:
        config["model_params"]["num_measured"] = 1

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
