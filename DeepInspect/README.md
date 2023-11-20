# DeepInspect
A simplified Pytorch implement of "DeepInspect: A Black-box Trojan Detection and Mitigation Framework for Deep Neural Networks, IJCAI 2019".  

The paper propose four main steps: Model Inversion, Trigger Generation, Anomaly Detection and Model Patching to detect and mitigate backdoor attacks. In this repository, we implement Trigger Generation and Model Patching process to mitigate backdoor attacks, leaving Model Inversion and Anomaly Detection to oracle. Besides, I remove $L_{GAN}$ because I don't know what $D\_{prob}$ is when $D$ is a fixed pre-trained model. 

## Environment Setups

- python 3.8.16
- torch 1.13.1
- torchvision 0.14.0
- numpy 1.21.5
- matplotlib 3.5.1

## Make backdoored models

We make backdoored models with Backdoor Toolbox by vtu81. I modified this repository to support MNIST dataset. My Backdoor Toolbox code link is https://github.com/123000001212/backdoor-toolbox. The offical Backdoor Toolbox code link is https://github.com/vtu81/backdoor-toolbox.

We offer trained bakcdoor models on MNIST and GTSRB dataset with BadNet attack in "backdoor_models/" folder.

## Running the Code

Remember to put trained backdoor models to "backdoor_models/" before running.

```
python main.py -dataset=mnist 
python main.py -dataset=gtsrb -clean_budget=5000
```
## Results

Example of results are presented at mnist-badnet.log and gtsrb-badnet.log.
