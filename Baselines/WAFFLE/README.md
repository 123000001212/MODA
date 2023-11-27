# WAFFLE
A simple Pytorch implement of "WAFFLE: Watermarking in Federated Learning" at 2021 40th International Symposium on Reliable Distributed Systems (SRDS). The official code link is https://github.com/ssg-research/WAFFLE. 

## Environment Setup

Our code successfully runs in Python 3.8.12 environment with torch==1.10.1, torchvision==0.11.2, numpy==1.21.5 and matplotlib==3.5.1. Make sure you meet these requirements before running the code.

Datasets will be downloaded to "/home/data". Make sure this path exists or change the path in the code. 

## Running the Code

Configure number clients and rounds as arguments. We only support clients=2, 3 or 6. And 10 to 20 rounds is already enough for our code. Here are some example commands to run the code.

```
python WAFFLE-MNIST_nclients.py --clients=2 --rounds=20
python WAFFLE-MNIST_nclients.py --clients=3 --rounds=20
python WAFFLE-MNIST_nclients.py --clients=6 --rounds=20
python WAFFLE-CIFAR10_nclients.py --clients=2 --rounds=20
python WAFFLE-CIFAR10_nclients.py --clients=3 --rounds=20
python WAFFLE-CIFAR10_nclients.py --clients=6 --rounds=20
```

## Results

The code will print out lots of accuracy values. As shown in the example below, in each round, the values at top and bottom are Clean Test set Accuracy values and the middle ones at "Retain" lines are Watermark Accuracy values. 

```
-- Round 18 -- | Accuracy of model in test set is: 0.991000 -> Test Accuracy
-- Retain 18- 1 -- | Accuracy of model in test set is: 0.640000 -> Watermark Accuracy
-- Retain 18- 2 -- | Accuracy of model in test set is: 0.900000 -> Watermark Accuracy
-- Retain 18- 3 -- | Accuracy of model in test set is: 0.990000 -> Watermark Accuracy
Accuracy of model in test set is: 0.990600 -> Test Accuracy
```

Finally, the code will print both Clean Test set Accuracy and Watermark Accuracy after the last round. 

```
-- Task Acc --
Accuracy of model in test set is: 0.991000
-- WaterMark Acc --
Accuracy of model in test set is: 0.990000
```

