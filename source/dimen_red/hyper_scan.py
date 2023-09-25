import os 

import optuna
import argparse 
import yaml
import pickle


from math import ceil
import csv
from dimensionality_reduction import dimensionality_reduction


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Hyperparameter scan for dimensionality reduction')
    parser.add_argument('-s', '--sampler', 
                        type = str,
                       help = "Type of sampler", 
                        default = "grid" 
                       )
    
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/training.yaml')

    parser.add_argument('--gpu',  '-g',
                        dest = "gpu_num",
                        help='GPU number',
                        default= 2)
    
    args = parser.parse_args()
    gpu = args.gpu_num
    
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
      
    # Model parameters 
    
    sampler_type = args.sampler
    
    
    fieldnames = ['Trial', 'lr', 'batch_size', 'b1', 'train_loss', 'valid_loss']
    
    method = "autoencoder" 
    data = config['data']
    hp0 = config['hp']
    save_dir = config['save_dir'] 
    
    continue_study = False
#     load_root = "/afs/cern.ch/work/s/suchang/shared/Data/"
    load_root = "/data/suchang/shared/Data/"
    
    if not continue_study :     
        os.makedirs(save_dir) 
        with open(os.path.join(save_dir, 'Intermediate_summary.csv'), mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    
    
    def objective(trial): 
        hp = hp0.copy()
        
        batch_size = trial.suggest_int("batch_size", 128, 1024, log=True)
        lr = trial.suggest_float("lr", 0.0001, 0.01, log=True) 
        b1 = trial.suggest_float("b1", 0.5, 0.9) 
        
        hp['training_params']['batch_size'] = batch_size
        hp['optim_params']['lr'] = lr 
        hp['optim_params']['betas'] = (b1, hp['optim_params']['betas'][1])
        
        snapshot_dir = os.path.join(save_dir, "Trial_" + str(trial.number+1))
        os.makedirs(snapshot_dir) 
        
        f = open(os.path.join(snapshot_dir, "summary.txt"), "+w") 
        f.write("batch_size : " + str(batch_size) + "\nlr : " + str(lr) + "\nb1 : " + str(b1)) 
        f.close() 
            
        train_loss, valid_loss = dimensionality_reduction(load_root, data, method, hp, gpu = gpu, snapshot_dir = snapshot_dir)    

        with open(os.path.join(save_dir, 'Intermediate_summary.csv'), mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            to_write = {'Trial' : trial.number+1, 'lr': lr, 'batch_size' : batch_size, "b1": b1, 
                        "train_loss": train_loss, "valid_loss": valid_loss} 
            writer.writerow(to_write) 
        
        #dump study
        pickle.dump(study, open(os.path.join(save_dir, 'study.pkl'), 'wb'))
    
    
        return valid_loss
    
    
    


    if sampler_type == "grid" : 
        
        
        
        lr = [0.0001, 0.001, 0.01]
        b1 = [0.5, 0.6, 0.7, 0.8, 0.9]
        batch_size = [128, 256, 512, 1024]
        
        nb_trials = len(lr) * len(b1) * len(batch_size) 
        print(len(lr), len(b1), len(batch_size), nb_trials)
        search_space = {"lr": lr, "b1": b1, "batch_size" : batch_size}
        
        sampler = optuna.samplers.GridSampler(search_space)
    elif sampler_type == "cmaes" : 
        sampler = optuna.samplers.CmaEsSampler()
        nb_trials = 100
     
    
    
    if continue_study :
        study = pickle.load(open(os.path.join(save_dir, 'study.pkl'), 'rb'))
    else : 
        study = optuna.create_study(direction = "minimize", sampler = sampler)
    
    study.optimize(objective, n_trials = nb_trials) 
    
    
    
    # Final summary
    f= open(os.path.join(save_dir, "Summary.txt"),"w")

    f.write("Best Parameters:")
    f.write("\nRun Number: " + str(study.best_trial.number) + "   Accuracy: "+ str(study.best_trial.value) + 
            "   Parameters: "+ str(study.best_trial.params) +  "\n\n")
    f.write("All Runs:")
    for nb in range(len(study.trials)):
        f.write("\n" + str(study.trials[nb]))
    f.close()
    