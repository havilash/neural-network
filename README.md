<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Havilash/Neural-Network/edit/main/README.md">
    <img src="logo.svg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">PyCore</h3>

  <p align="center">
    Classification Neural Network
  </p>
</div>

## Description

This project is a versatile neural network package specifically designed for classification tasks. It can be used for training various deep learning models, including image recognition. The package includes an example project for MNIST classification to demonstrate its capabilities. With its flexible architecture, the package is suitable for a range of classification tasks requiring deep learning, offering high accuracy and reliability.

## Getting Started

### Dependencies

* numpy
* matplotlib
* albumentations
* scikit-learn
* jupyter
* scipy
* scikit-image
* pillow
* customtkinter

### Installing

1. Clone the repository
2. Navigate to the package directory
3. Run `pip install .` to install package

or 

1. Run `pip install git+https://github.com/Havilash/Neural-Network.git#egg=neural_network` to install package

### Executing program

* Import package
```python
from neural_network import activations, costs, nn as neural_network, layers, gui, constants
from neural_network.data import get_mnist_data, get_augmented_mnist_data, train_test_split
from neural_network.filters import ALL_FILTERS
```
* Run example project `python ./neural_network/main.py` or `python -m neural_network.main`

## Authors
 
* [@Havilash](https://github.com/Havilash)
* [@Gregory](https://github.com/rergr)
* [@Nicolas](https://github.com/Nic01asCT)
* [@Ensar](https://github.com/Ensar05)

## License

see the LICENSE file for details
