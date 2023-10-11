.. role:: raw-html-m2r(raw)
   :format: html


:raw-html-m2r:`<!--Back to the top -->`
:raw-html-m2r:`<a name="readme-top"></a>`


.. raw:: html

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


About the codes
---------------

This repository contains code for image classification using a hybrid approach that combines classical dimensionality reduction with a quantum classifier. The code is implemented in `PyTorch <link_to_pytorch>`_ for dimensionality reduction and in `Jax <https://github.com/google/jax>`_ and `PennyLane <https://github.com/PennyLaneAI/pennylane>`_ for quantum operations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   dimen_red
   quantum_classifier
   

Project Workflow
^^^^^^^^^^^^^^^^

The project consists of two main steps:


#. 
   **Classical dimensionality reduction** : Extract essential features from the input image data. 

#. 
   **Feauture classification with quantum classifier** : Utilize a quantum classifier for image classification. 

Currently supported features include:


* Dimensionality reduction methods

  * *PCA*
  * *Convolutional Autoencoder*

* Image dataset 

  * `\ *MNIST* <https://pytorch.org/vision/0.15/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST>`_
  * `\ *EuroSAT* <https://github.com/phelber/EuroSAT>`_

Prerequisites
-------------

Before running the code, make sure to install the required packages by running:

.. code-block::

   pip install -r requirements.txt

Additionally, install ``jaxlib`` compatible with your CUDA version using the following command:

.. code-block::

   pip install --upgrade jaxlib==0.4.7+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

For more detailed instructions on Jax installation, please refer to the officia `Jax Installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_. 


.. raw:: html

   <p align="right">(<a href="#readme-top">back to top</a>)</p>


Usage
-----

Use the following commands  :


#. To perform dimensionality reduction of the images. 

.. code-block::

   python run_dim_red.py --gpu gpu_num --config config_file_name


#. To train the quantum classifier using the reduced features already stored in a file. 

.. code-block::

   python run_qc_training.py --gpu gpu_num --config config_file_name


#. To run the whole training pipeline. 

.. code-block::

   python run.py --gpu gpu_num --config config_file_name

How to structure the configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The configuration file ``configs/training.yaml`` should be structured as follows : 


* **data**\ : Name of dataset on which we perform the feature extraction. 
* **method**\ :  Dimensionality reduction method for dimensionality reduction. Currently, only PCA (\ ``pca``\ ) and convolutional autoencoder (\ ``ae``\ ) are supported.
* **num_components** : Dimension of extracted features. 
* 
  **load_root**\ : Root directory containing 


  #. in case of performing dimensionality reduction : the original images. 
  #. in case of performing only the feature classification :  the reduced features.

* 
  **dimensionality_reduction**\ *: The hyperparameters required in case of ``method == ae``.


  * **training_params**\ : 

    * *num_epohcs* : Number of training epochs. 
    * *batchsize* : Training batch size.

  * **optim_params**\ : 

    * *lr* : Learning rate. 
    * *betas* : $[\beta_1, \beta_2]$ value of the Adam optimizer. $[0.9,  0.999]$ by default. 


* **quantum_classifier**\ : The hyperparameters required for the quantum classifier training.

  * **training_params**\ : 

    * *num_epohcs* : Number of training epochs. 
    * *batchsize* : Training batch size

  * **model_params** :   

    * *num_wires* : Number of qubits in the quantum classifier.  
    * *ver* :  Quantum circuit architecture version.  
    * *Embedding* :  Quantum embedding method.  
    * *trans_inv* : Boolean to indicate whether the mdel is constructed in a translational invariant way.  

  * **opt_params**\ : 

    * *lr* : Learning rate. 
    * *b1* : $\beta_1$ value of the Adam optimizer. 0.9 by default. 
    * *b2* : $\beta_2$ value of the Adam optimizer. 0.999 by default.

For more details, check the config files ``configs/config_pca.yaml`` and ``configs/config_ae.yaml``. 


.. raw:: html

   <p align="right">(<a href="#readme-top">back to top</a>)</p>


License
-------

Distributed under the Apache License. See ``LICENSE.txt`` for more information.


.. raw:: html

   <p align="right">(<a href="#readme-top">back to top</a>)</p>


:raw-html-m2r:`<!-- CONTACT -->`

Contact
-------

Su Yeon Chang - `@SyChang97 <https://twitter.com/SyChang97>`_ - su.yeon.chang@cern.ch

Project Link: `https://github.com/sychang42/QML_image_classification <https://github.com/sychang42/QML_image_classification>`_


.. raw:: html

   <p align="right">(<a href="#readme-top">back to top</a>)</p>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
