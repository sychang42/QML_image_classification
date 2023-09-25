import argparse
import json
import os

import yaml
from dimensionality_reduction import dimensionality_reduction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Peform dimensionality reduction"
    )

    parser.add_argument(
        "-s", "--sampler", type=str, help="Type of sampler", default="grid"
    )

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

    # Model parameters

    method = "autoencoder"
    data = config["data"]
    hp = config["hp"]
    save_dir = config["save_dir"]

    load_root = "/data/suchang/shared/Data/"

    snapshot_dir = None
    if save_dir is not None:
        count = 1
        while os.path.exists(os.path.join(save_dir, "run" + str(count))):
            count += 1
        snapshot_dir = os.path.join(save_dir, "run" + str(count))
        os.makedirs(snapshot_dir)

        json.dump(config, open(os.path.join(snapshot_dir, "config.json"), "+w"))

    train_loss, valid_loss = dimensionality_reduction(
        load_root, data, method, hp, gpu=gpu, snapshot_dir=snapshot_dir
    )
