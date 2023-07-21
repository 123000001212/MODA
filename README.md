# MODA
 Pytorch implement of MODA

## Requirements

- python 3.8.12
- torch 1.10.1
- torchvision 0.11.2
- torchsummary 1.5.1
- numpy 1.21.5
- matplotlib 3.5.1

## Descriptions

- main.ipynb: Asynchronous Federated Learning procedure where an adversary trains an ACGAN to generate inversed images.
- wm_gen.ipynb: Generate inversed images from the generator of ACGAN.
- unlearning.ipynb: Unlearning victim's watermark by generated images containing watermark pattens.
- dataset_init.py: Initialize MNIST, CIFAR10, FashionMNIST, SVHN, and GTSRB dataset.
- model_init.py: Define different model structures.

## Running attacks

Step 1: Train ACGAN

~~~
run main.ipynb
~~~

Step 2: Generate inversed images

~~~
run wm_gen.ipynb
~~~

Step3: Attacks

~~~
run unlearning.ipynb
~~~

## Other Related Code Repositories

- WAFFLE: https://github.com/123000001212/WAFFLE
- NC: https://github.com/123000001212/backdoor-toolbox
- DeepInspect: https://github.com/123000001212/DeepInspect


