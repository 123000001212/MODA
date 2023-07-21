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

- Modify all absolute paths in our code (e.g. "dataset_init.py") to the correct paths on your machine before running.

##### Step 1: Train ACGAN

Stop when all watermark reach a high accuracy and "results/fake-image-x.png" show inversed triggers.

~~~
run train_ACGAN.ipynb
~~~

##### Step 2: Generate inversed images

Modify the first row in the last cell to `for i in [x,x,x]:`, where x,x,x are the target labels of user watermarks. (DO NOT include the target label of adversary's watermark here).

~~~
run wm_gen.ipynb
~~~

##### Step 3: Correct labels of inversed trigger images

Inspect corresponding watermark data images and its' target label. Modify X,Y,Z in 3 lines in the corresponding  cell: 

- if i==X:

- wmdata.append((data,Y))

- torch.save(wmdataset,"./inversed_wm_data/{DATASET}_{NUM_USERS}/userZ.pth")

  ,where X is the order of label in folder "wm_{DATASET}" (start from 0), Y is the original label of images, Z is the serial number of users.

Run the corresponding cell each time after you modify, until all label in "wm_{DATASET}" folder are processed.

```
run correct_label.ipynb
```

##### Step 4: Attacks via unlearning

~~~
run unlearning.ipynb
~~~

NOTE: The experimental results of attack success rate and classification accuracy can be found in unlearning.ipynb.

## Other Related Code Repositories

- WAFFLE: https://github.com/123000001212/WAFFLE
- NC: https://github.com/123000001212/backdoor-toolbox
- DeepInspect: https://github.com/123000001212/DeepInspect


