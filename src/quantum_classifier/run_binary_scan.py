import os
import sys
sys.path.append(os.path.dirname(__file__)) 

import yaml
import argparse


from dataset import load_data
from models.train import train
import json 
from math import ceil, log2
import numpy as np


if __name__ == "__main__":
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    # Retrieve argument from yaml file
    parser = argparse.ArgumentParser(
        description="Run qGAN experiments on simulator")
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/training.yaml')
    parser.add_argument('--gpu',  '-g',
                        dest = "gpu_num",
                        help='GPU number',
                        default="0")
    
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    
    data = config['training_params']['data']
    n_components = config['training_params']['n_components'] 
    classes = [8,9]
    if "BCE_loss" in config['training_params']['loss_type'] : 
        config['model_params']['num_measured'] = int(ceil(log2(len(classes))))
    else : 
        config['model_params']['num_measured'] = 1
    
    root0 = config['logging_params']['save_dir']  # Directory to save
    
    # Load data    
    load_dir = '/data/suchang/sy_phd/v4_QML_EO/v1_classification/data/' 
        
      
    num_repeat = 20 
    methods = ['pca', 'ae', 'deepae', 'verydeepae']
    for method in methods : 
        root = os.path.join(root0, method) 
        for i in range(3, 10) : 
            for j in range(i+1, 10) : 
                classes = [i,j]
                X_train, Y_train, X_test, Y_test = load_data(load_dir, data, method, n_components, classes)
                train_ds = {'image' : X_train, "label" : Y_train}
                test_ds = {'image' : X_test, "label" : Y_test}
                config['training_params']['classes'] = classes 
                config['training_params']['method'] = method
                save_dir = os.path.join(root, str(i) + str(j)) 
                
                for runID in range(num_repeat) : 
                    seed = np.random.randint(10000)
                    config["training_params"]['seed'] = seed 

                    snapshot_dir = None 
                    if save_dir is not None : 
                    # Create folder to save training results
                        path_count = 1
                        snapshot_dir = os.path.join(save_dir, 'run' + str(path_count))
                        while os.path.exists(snapshot_dir):
                            path_count += 1
                            snapshot_dir = os.path.join(save_dir, 'run' + str(path_count))

                        os.makedirs(snapshot_dir)
                        json.dump(config, open(os.path.join(snapshot_dir, "summary.json"), "+w"))


                    train(train_ds, test_ds, config['training_params'], config['model_params'], config['optim_params'], 
                          snapshot_dir)