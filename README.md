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

- train_ACGAN.py: Asynchronous Federated Learning procedure where an adversary trains an ACGAN to generate inversed images.
- gen_inversed_wm.ipynb: Generate inversed images by the generator of ACGAN.
- moda.py: Unlearning victims' watermarks.
- experiments.ipynb: Unlearning victims' watermarks and other baselines.
- dataset_init.py: Initialize MNIST, CIFAR10, FashionMNIST, SVHN, and GTSRB datasets.
- model_init.py: Define different model structures.

## Running attacks

#### Step 0: Initialization

Modify all absolute paths in our code (e.g. "dataset_init.py") to the correct paths on your machine before running.

#### Step 1: Train ACGAN

~~~
python train_acgan.py --dataset="MNIST" --num_users=3
~~~

Stop when all watermark reach a high accuracy and "results/fake-image-x.png" show inversed triggers.

#### Step 2: Generate inversed watermarks

~~~
run gen_inversed_wm.ipynb
~~~

#### Step 3: Attacks via unlearning

~~~
python moda.py --dataset="MNIST" --num_users=3
~~~

NOTE: The experimental results of attack success rate and classification accuracy can be found in experiments.ipynb.

## Other Related Code Repositories

- WAFFLE: https://github.com/123000001212/WAFFLE
- NC: https://github.com/123000001212/backdoor-toolbox
- DeepInspect: https://github.com/123000001212/DeepInspect


These codes can be found in this repository as well (in "Baselines" folder). Results of these codes are presented at experiments.ipynb. 

## Citation
```
@article{zhang2023moda,
  title={MODA: model ownership deprivation attack in asynchronous federated learning},
  author={Zhang, Xiaoyu and Lin, Shen and Chen, Chao and Chen, Xiaofeng},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2023},
  publisher={IEEE}
}
```
