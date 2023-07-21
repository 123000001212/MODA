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

- train_ACGAN.ipynb: Asynchronous Federated Learning procedure where an adversary trains an ACGAN to generate inversed images.
- wm_gen.ipynb: Generate inversed images by the generator of ACGAN.
- unlearning.ipynb: Unlearning victims' watermarks.
- dataset_init.py: Initialize MNIST, CIFAR10, FashionMNIST, SVHN, and GTSRB datasets.
- model_init.py: Define different model structures.

## Running attacks

Step 1: Train ACGAN

~~~
run train_ACGAN.ipynb
~~~

Step 2: Generate inversed images

~~~
run wm_gen.ipynb
~~~

Step3: Attacks via unlearning

~~~
run unlearning.ipynb
~~~

## Other Related Code Repositories

- WAFFLE: https://github.com/123000001212/WAFFLE
- NC: https://github.com/123000001212/backdoor-toolbox
- DeepInspect: https://github.com/123000001212/DeepInspect


