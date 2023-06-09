# QML_image_classification

`run.py --gpu gpu_num --config config_file_name`

### How to use config file 
data: MNIST  

hp: 
  training_params: 
    num_epoch: 100 
    batch_size: 1024
  model_params: 
    nz: 16 
  optim_params: 
    lr: 0.001
    betas: [0.9, 0.999]

save_dir: ../Result/hyper_scan/MNIST


data = Data to use
nz = Latent space dimension
lr, betas = learning rate for adam optimizer 