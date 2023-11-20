#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--clients', type=int, default=2, choices=[2,3,6])
parser.add_argument('--rounds', type=int, default=20)
args = parser.parse_args()

# In[2]:


# 定义MINST_L5模型 / Definition of MNIST_L5 model
class MNIST_L5(nn.Module):
    def __init__(self, dropout=0.0):
        super(MNIST_L5, self).__init__()

        self.dropout = dropout

        self.block = nn.Sequential(
            nn.Conv2d(1, 32, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(128 * 5**2 , 200)
        self.fc2 = nn.Linear(200, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        out = self.block(x)
        out = out.view(-1,  128 * 5**2)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return F.log_softmax(out,1)


# In[3]:


# 加载MINST数据集 / Load and split MNIST dataset
trainset = torchvision.datasets.MNIST(root = '/home/data',train = True,
                transform = torchvision.transforms.ToTensor(),download = True)
datasets = torch.utils.data.random_split(trainset, [int(60000//args.clients) for i in range(args.clients)])

testset=torchvision.datasets.MNIST(root = '/home/data',train = False,
                transform = torchvision.transforms.ToTensor(),download = True)
testloader = torch.utils.data.DataLoader(testset, batch_size = 50, shuffle = False, num_workers = 0)


# In[4]:


# 加载watermarked mnist数据集 / Load watermarked mnist dataset
wmtransform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像 / Color image to Grayscale
    torchvision.transforms.ToTensor()
])
watermarked_mnist = torchvision.datasets.ImageFolder(root='./data/MWAFFLE',transform=wmtransform)
MWAFFLE_Loader=torch.utils.data.DataLoader(watermarked_mnist, batch_size =50, shuffle = True, num_workers = 0)

def imShow(img):
    img = img
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
for i, data in enumerate(MWAFFLE_Loader, 0):
    inputs, labels = data
    imShow(torchvision.utils.make_grid(inputs))
    break
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    imShow(torchvision.utils.make_grid(inputs))
    break



# In[5]:


# 定义client类
class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, global_parameters):
        '''
            param: localEpoch 当前Client的迭代次数
            param: localBatchSize 当前Client的batchsize大小
            param: Net Server共享的模型
            param: LossFun 损失函数
            param: opti 优化函数
            param: global_parmeters 当前通讯中的全局参数
            return: 返回当前Client基于自己的数据训练得到的新的模型参数
        '''
        # 加载当前通信中最新全局参数 / Load the latest global parameters in the current communication
        # 传入网络模型，并加载global_parameters参数 / Load global_parameters
        Net.load_state_dict(global_parameters, strict=True)
        # 加载本地数据 / Load local data
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        # 优化器 / optimizer
        opti = torch.optim.SGD(params=Net.parameters(), lr=0.1)
        # 设置迭代次数 / Set the number of iterations
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                # 加载到GPU上 / Load to GPU
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据 / Input data
                preds = Net(data)
                # 计算损失函数 / Calculate the Loss
                loss = lossFun(preds, label)
                # 反向传播 / Back Propagation
                loss.backward()
                # 计算梯度，并更新梯度 / Update gradients
                opti.step()
                # 将梯度归零，初始化梯度 / Zero gradients
                opti.zero_grad()
        # 返回当前Client基于自己的数据训练得到的新的模型参数 / Returns the new model parameters trained by the current client based on their own data
        return Net.state_dict()

    def local_val(self):
        pass


# In[6]:


# 创建client / Create clients
clients=[client(datasets[i],'cuda:0') for i in range(args.clients)]
# 全局模型 / global net
global_net=MNIST_L5().cuda()
share_net=MNIST_L5().cuda()
# 损失函数 / loss function
criterion = nn.CrossEntropyLoss()


# In[7]:


# 测试模型的准确率 / Testing the accuracy of the model on data_loader (test or watermark)
def test_func(model,data_loader):
    model.eval()
    data_cnt = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            _,predict = torch.max(model.forward(data).data,1)  # 取最大值的索引为预测结果 / The index with the highest value is the predicted result
            correct += int(torch.sum(predict==target).cpu().numpy())  # 统计正确个数 / Count the correct number
            data_cnt += len(target)
    print('Accuracy of model in test set is: %f'%(correct/data_cnt))
    return correct/data_cnt

# test_func(global_net,testloader)


# In[8]:


# mnist pretrain
def pretrain():
    global_net.train()
    lossFun=nn.CrossEntropyLoss()
    opti = torch.optim.SGD(global_net.parameters(), lr=0.1, momentum=0.5, weight_decay=5e-05)
    for r in range(25):
        for data, label in MWAFFLE_Loader:
            data, label = data.to('cuda:0'), label.to('cuda:0')
            preds = global_net(data)
            loss = lossFun(preds, label)
            loss.backward()
            opti.step()
            opti.zero_grad()
        print('-- Pretrain %2d -- | '%(r+1),end='')
        acc=test_func(global_net,MWAFFLE_Loader)


# In[9]:


# mnist retrain
def retain(i):
    global_net.train()
    lossFun=nn.CrossEntropyLoss()
    opti = torch.optim.SGD(global_net.parameters(), lr=0.005)
    for r in range(100):
        for data, label in MWAFFLE_Loader:
            data, label = data.to('cuda:0'), label.to('cuda:0')
            preds = global_net(data)
            loss = lossFun(preds, label)
            loss.backward()
            opti.step()
            opti.zero_grad()
        print('-- Retain %2d-%2d -- | '%(i+1,r+1),end='')
        acc=test_func(global_net,MWAFFLE_Loader)
        if acc>=0.98:
            break


# In[10]:


# 分布式训练过程 / Distributed training process
def train():
    pretrain()
    global_parameters=global_net.state_dict()
    # 交互轮数 / Number of interactive rounds
    for r in range(args.rounds):
        print('-- Round %2d -- | '%(r+1),end='')
        choice=[clients[i] for i in range(args.clients)] # n方 / n participants
        num_clients=args.clients # 记得改这里
        
        sum_parameters = None
        for c in choice:
            # 每个client训练 / Each client trains an updates
            local_parameters = c.localUpdate(localEpoch=10,localBatchSize=50,Net=share_net,lossFun=criterion,
                                                  global_parameters=global_parameters)
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        # 取平均值，得到本次通信中Server得到的更新后的模型参数 / Take the average value to obtain the updated model parameters obtained by the server in this communication
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_clients)
    
        # 每轮训练后加载模型参数 / Load global model parameters after each training round
        global_net.load_state_dict(global_parameters, strict=True)
        # 测试 / test clean accuracy
        test_func(global_net,testloader)
        retain(r)
        test_func(global_net,testloader)


# In[11]:


train()


# In[12]:


print('-- Task Acc --')
test_func(global_net,testloader)
print('-- WaterMark Acc --')
test_func(global_net,MWAFFLE_Loader)

