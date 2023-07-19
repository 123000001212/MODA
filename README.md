# MODA
 Pytorch implement of MODA

## Environment Setups

- python 3.8.12
- torch 1.10.1
- torchvision 0.11.2
- torchsummary 1.5.1
- numpy 1.21.5
- matplotlib 3.5.1

## Data Preparation



## main.ipynb
Asynchronous Federated Learning procedure where an adversary trains an ACGAN to generate inversed images.
## wm_gen.ipynb
Generate inversed images from the generator of ACGAN.
## unlearning.ipynb
Unlearning victim's watermark by generated images containing watermark pattens.
## dataset_init.py
Initialize MNIST, CIFAR10, FashionMNIST, SVHN, GTSRB, watermark and unlearning datasets.
## model_init.py
Define different model structures for different datasets.
