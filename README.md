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

You can also run the code on [Google Colab](https://colab.research.google.com/drive/1ZuPgz3mkVcIMVG9BZ0UCpw7PhfHQXHXu?usp=sharing). 

### Project Workflow

The project consists of two main steps:

1. __Classical dimensionality reduction__ : Extract essential features from the input image data. 

2. __Feauture classification with quantum classifier__ : Utilize a quantum classifier for image classification. 


Currently supported features include:
- Dimensionality reduction methods
    - *PCA* (``pca``)
    - *Convolutional Autoencoder* (``ae``)
- Image dataset 
    - [*MNIST*](https://pytorch.org/vision/0.15/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST)
    - [*FashionMNIST*](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)
    - [*EuroSAT*](https://github.com/phelber/EuroSAT)
    - [*SAT4*](https://csc.lsu.edu/~saikat/deepsat/)

## Prerequisites
Before running the code, make sure to install the required packages by running:

```
pip install -r requirements.txt
```

Additionally, install `jaxlib` compatible with your CUDA version using the following command:

```
pip install --upgrade jaxlib==0.4.7+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For more detailed instructions on Jax installation, please refer to the officia [Jax Installation guide](https://jax.readthedocs.io/en/latest/installation.html). 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Usage

Use the following commands  :

1. To perform dimensionality reduction of the images. 

```
python run_dim_red.py --gpu gpu_num --config config_file_name
```

2. To train the quantum classifier using the reduced features already stored in a file. 

```
python run_qc_training.py --gpu gpu_num --config config_file_name
```


3. To run the whole training pipeline. 

```
python run.py --gpu gpu_num --config config_file_name
```

### How to structure the configuration file

The configuration file `configs/training.yaml` should be structured as follows : 

* __data__: Name of dataset on which we perform the feature extraction. 
* __method__:  Dimensionality reduction method for dimensionality reduction. Currently, only PCA (``pca``) and convolutional autoencoder (``ae``) are supported.
* __num_components__ : Dimension of extracted features. 
* __load_root__: Root directory containing 
  1. in case of performing dimensionality reduction : the original images. 
  2. in case of performing only the feature classification :  the reduced features.

* __dimensionality_reduction__*: The hyperparameters required in case of ``method == ae``.
  - __training_params__: 
    - *num_epohcs* : Number of training epochs. 
    - *batchsize* : Training batch size.
  - __optim_params__: 
    - *lr* : Learning rate. 
    - *betas* : $[\beta_1, \beta_2]$ value of the Adam optimizer. $[0.9,  0.999]$ by default. 


* __quantum_classifier__: The hyperparameters required for the quantum classifier training.
  - __training_params__: 
    - *num_epohcs* : Number of training epochs. 
    - *batchsize* : Training batch size
  - __model_params__ :   
    - *num_wires* : Number of qubits in the quantum classifier.  
    - *ver* :  Quantum circuit architecture version.  
    - *Embedding* :  Quantum embedding method.  
    - *trans_inv* : Boolean to indicate whether the mdel is constructed in a translational invariant way.  
  - __opt_params__: 
    - *lr* : Learning rate. 
    - *b1* : $\beta_1$ value of the Adam optimizer. 0.9 by default. 
    - *b2* : $\beta_2$ value of the Adam optimizer. 0.999 by default.

For more details, check the config files ``configs/config_pca.yaml`` and ``configs/config_ae.yaml``. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## License

Distributed under the Apache License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Su Yeon Chang - [@SyChang97](https://twitter.com/SyChang97) - su.yeon.chang@cern.ch

Project Link: [https://github.com/sychang42/QML_image_classification](https://github.com/sychang42/QML_image_classification)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


