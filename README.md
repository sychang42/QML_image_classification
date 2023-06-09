# QML_image_classification

`run.py --gpu gpu_num --config config_file_name`

### How to use config file 
data: MNIST  <br />

hp: <br />
  training_params: <br />
    num_epoch: 100 <br />
    batch_size: 1024<br />
  model_params: <br />
    nz: 16 <br />
  optim_params: <br />
    lr: 0.001<br />
    betas: [0.9, 0.999]<br />
<br />
save_dir: ../Result/hyper_scan/MNIST<br />


data = Data to use <br />
nz = Latent space dimension <br /> 
lr, betas = learning rate for adam optimizer <br /> 