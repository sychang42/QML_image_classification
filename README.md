<!--Back to the top -->
<a name="readme-top"></a>


<div align="center">
<h3 align="center">Practical Quantum Machine Learning for Image Classification</h3>
  <p align="center">
     Image classification using a hybrid approach with classical dimensionality reduction and a quantum classifier.
    <br />
    <a href="https://github.com/github_username/repo_name"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    ·
    <a href="https://github.com/sychang42/QML_image_classification/issues">Report Bug</a>
    ·
    <a href="https://github.com/sychang42/QML_image_classification/issues">Request Feature</a>
  </p>
</div>


## About the codes

This repository contains code for image classification using a hybrid approach that combines classical dimensionality reduction with a quantum classifier. The code is implemented in [PyTorch](link_to_pytorch) for dimensionality reduction and in [Jax](https://github.com/google/jax) and [PennyLane](https://github.com/PennyLaneAI/pennylane) for quantum operations.

### Project Workflow

The project consists of two main steps:

1. __Classical dimensionality reduction__ : Extract essential features from the input image data. 

2. __Feauture classification with quantum classifier__ : Utilize a quantum classifier for image classification. 


Currently supported features include:
- Dimensionality reduction methods
    - *PCA*
    - *Convolutional Autoencoder*
- Image dataset 
    - [*MNIST*](https://pytorch.org/vision/0.15/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST)
    - [*EuroSAT*](https://github.com/phelber/EuroSAT)

## Prerequisites
Before running the code, make sure to install the required packages by running:

```
pip install -r requirements.txt
```

Additionally, install `jaxlib` compatible with your CUDA version using the following command:

```
pip install --upgrade jax==0.4.7+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For more detailed instructions on Jax installation, please refer to the officia [Jax Installation guide](https://jax.readthedocs.io/en/latest/installation.html). 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Usage

To train the model, use the following command : 

```
python run.py --gpu gpu_num --config config_file_name
```

The configuration file `configs/training.yaml` should be structured as follows : 

* __dataset_params__: 
  - *data* : Name of training dataset. 
  - *img_size* : Input image size.
  - *classes* : List of integers representing data classes to be trained. Currently, only binary classification is implemented.
  - *n_data* : Number of training samples. Set to `Null` to use the entire dataset.

* __training_params__: 
  - *num_epohcs* : Number of training epochs. 
  - *batchsize* : Training batch size
  - *loss_type* : Type of loss function used to train the model. Currently, only Binary Cross-Entropy (BCE) loss is implemented.
  
* __model_params__: 
  - *num_wires* : Number of qubits in the quantum classifier. 
  - *equiv* : Boolean to indicate whether an equivariant neural network is used. 
  - *trans_inv* : Boolean to indicate whether the model is constructed in a translational invariant way. 
  - *ver* :  Quantum circuit architecture version. 
  - *symmetry_breaking* : Boolean to indicate whether $RZ$ gates at the end of the quantum circuit in case of the *EquivQCNN*. 
  - *delta* : Range of uniform distribution from which the initial parameters are sampled.  
  
* __opt_params__: 
  - *lr* : Learning rate. 
  - *b1* : $\beta_1$ value of the Adam optimizer. 0.9 by default. 
  - *b2* : $\beta_2$ value of the Adam optimizer. 0.999 by default.

* __logging_params__: 
  - *save_dir* : Root directory to store the training results. If `Null`, the results will not stored.



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

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## License

Distributed under the Apache License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Su Yeon Chang - [@SyChang97](https://twitter.com/SyChang97) - su.yeon.chang@cern.ch

Project Link: [https://github.com/sychang42/QML_image_classification](https://github.com/sychang42/QML_image_classification)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


