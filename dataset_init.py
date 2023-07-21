import copy
import os
from matplotlib.transforms import Transform


import torch
from numpy.random import randint
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class wmDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform=transform
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        self.transform=None
        if self.transform != None:
            image = self.transform(image)
        return image, label

class myDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]

        return image, label

class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, txt, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('RGB')  # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)  # 数据标签转换为Tensor
        #label = torch.from_numpy(np.array(label))
        return img, label  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

def dataset_init(dataset):
    if dataset == 'MNIST':
        dataTransform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5 ))])
        clean_dataset = datasets.MNIST(root='/home/data/', train=True, transform=dataTransform)
        test_dataset = datasets.MNIST(root='/home/data/', train=False, transform=dataTransform)
    if dataset == 'SVHN':
        dataTransform = transforms.Compose([
                                    transforms.CenterCrop(32),
                                    transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
        clean_dataset = datasets.SVHN(root='/home/data/SVHN/', split='train', transform=dataTransform)
        test_dataset = datasets.SVHN(root='/home/data/SVHN/', split='test', transform=dataTransform)

    if dataset == 'FashionMNIST':
        dataTransform = transforms.Compose(
                                    [transforms.Resize(size=(32, 32), interpolation=InterpolationMode.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
                                    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
                                    ])
        clean_dataset = datasets.FashionMNIST('/home/data/', train=True, transform=dataTransform)
        test_dataset = datasets.FashionMNIST('/home/data/', train=False, transform=dataTransform)
    if dataset == 'GTSRB':
        dataTransform = transforms.Compose([
                                        transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
        train_dataset_root = '/home/data/GTSRB/Final_Training/Images'
        clean_dataset = datasets.ImageFolder(train_dataset_root, transform=dataTransform)
        test_dataset = MyDataset(txt='/home/data/GTSRB/Final_Test/test_dataset.txt', transform=dataTransform)
    if dataset == 'CIFAR10':
        dataTransform = transforms.Compose([
                            transforms.CenterCrop(32),
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
        clean_dataset = datasets.CIFAR10('/home/data/', train=True, transform=dataTransform)
        test_dataset = datasets.CIFAR10('/home/data/', train=False, transform=dataTransform)
    user_wm_dataset,  train_dataset= [], copy.deepcopy(clean_dataset)

    for root, dirs, files in os.walk('/home/zcy/MODA/wm_data/' + dataset + '/'): 
        files.sort()
        for file in files:
            user_wm_dataset.append(wmDataset(torch.load(root + file)))
    for i in range(len(user_wm_dataset)):
        train_dataset = train_dataset + user_wm_dataset[i]
    return clean_dataset, train_dataset, user_wm_dataset, test_dataset

def dataset_init_10users(dataset):
    if dataset == 'MNIST':
        dataTransform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5 ))])
        wmdataTransform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5 ))])
        clean_dataset = datasets.MNIST(root='/home/data/', train=True, transform=dataTransform)
        test_dataset = datasets.MNIST(root='/home/data/', train=False, transform=dataTransform)
    
    user_wm_dataset,  train_dataset= [], copy.deepcopy(clean_dataset)
    wm_dataset=datasets.ImageFolder("/home/zcy/MODA/wm_data_10_users/waffle_patten", transform=wmdataTransform)
    for index in range(10):
        wmdata=[]
        for data,i in wm_dataset:
            if i==index:
                wmdata.append((data,i))
        user_wm_dataset.append(wmDataset(wmdata))

    for i in range(len(user_wm_dataset)):
        train_dataset = train_dataset + user_wm_dataset[i]
    return clean_dataset, train_dataset, user_wm_dataset, test_dataset

def unlearning_dataset_init(dataset, num_users):
    if dataset == 'MNIST':
        dataTransform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5))])
        datatransform =  transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5))])
        clean_dataset = datasets.MNIST(root='/home/data/', train=True, transform=dataTransform)
        test_dataset = datasets.MNIST(root='/home/data/', train=False, transform=dataTransform)
    if dataset == 'SVHN':
        dataTransform = transforms.Compose([
                                    transforms.CenterCrop(32),
                                    transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
        clean_dataset = datasets.SVHN(root='/home/data/SVHN/', split='train', transform=dataTransform)
        test_dataset = datasets.SVHN(root='/home/data/SVHN/', split='test', transform=dataTransform)
    if dataset == 'FashionMNIST':
        dataTransform = transforms.Compose(
                                    [transforms.Resize(size=(32, 32), interpolation=InterpolationMode.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
                                    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
                                    ])
        clean_dataset = datasets.FashionMNIST('/home/data/', train=True, transform=dataTransform)
        test_dataset = datasets.FashionMNIST('/home/data/', train=False, transform=dataTransform)
    if dataset == 'GTSRB':
        dataTransform = transforms.Compose([
                                        transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
        train_dataset_root = '/home/data/GTSRB/Final_Training/Images'
        clean_dataset = datasets.ImageFolder(train_dataset_root, transform=dataTransform)
        test_dataset = MyDataset(txt='/home/data/GTSRB/Final_Test/test_dataset.txt', transform=dataTransform)
    if dataset == 'CIFAR10':
        dataTransform = transforms.Compose([
                            transforms.CenterCrop(32),
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
        clean_dataset = datasets.CIFAR10('/home/data/', train=True, transform=dataTransform)
        test_dataset = datasets.CIFAR10('/home/data/', train=False, transform=dataTransform)
    user_wm_dataset = []
    for root, dirs, files in os.walk('/home/zcy/MODA/wm_data_6_users/' + dataset + '/'): 
        files.sort()
        for file in files:
            user_wm_dataset.append(wmDataset(torch.load(root + file)))
    if num_users==3:
        adv_mi_dataset = wmDataset(torch.load('/home/zcy/MODA/inversed_wm_data/' + dataset + '/unlearn_trigger.pth'), transform=dataTransform) + wmDataset(torch.load('/home/zcy/MODA/inversed_wm_data/' + dataset + '/unlearn_unrelated.pth'), transform=dataTransform)
    elif num_users==2:
        user_wm_dataset=user_wm_dataset[:-1]
        adv_mi_dataset = wmDataset(torch.load('/home/zcy/MODA/inversed_wm_data/' + dataset + '/unlearn_trigger.pth'), transform=dataTransform)
    else:
        datatransform = None
        for root, dirs, files in os.walk('/home/zcy/MODA/inversed_wm_data/CIFAR10_6/'):
            adv_mi_dataset = wmDataset(torch.load(root + files[0]), transform=datatransform)
            for file in files[1:]:
                adv_mi_dataset += wmDataset(torch.load(root + file), transform=datatransform)

        
    unlearning_dataset = Subset(clean_dataset, randint(0, len(clean_dataset),size=int(len(clean_dataset)/num_users))) + user_wm_dataset[-1] + adv_mi_dataset

    return clean_dataset, unlearning_dataset, user_wm_dataset, test_dataset

def unlearning_dataset_init_10users(dataset, num_users=10,num_adv=1):
    if dataset == 'MNIST':
        dataTransform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5))])
        wmdataTransform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5 ))])
        clean_dataset = datasets.MNIST(root='/home/data/', train=True, transform=dataTransform)
        test_dataset = datasets.MNIST(root='/home/data/', train=False, transform=dataTransform)
    user_wm_dataset = []
    wm_dataset=datasets.ImageFolder("/home/zcy/MODA/wm_data_10_users/waffle_patten", transform=wmdataTransform)
    for index in range(10):
        wmdata=[]
        for data,i in wm_dataset:
            if i==index:
                wmdata.append((data,i))
        user_wm_dataset.append(wmDataset(wmdata))
    if num_users==10:
        datatransform = None
        for root, dirs, files in os.walk('/home/zcy/MODA/inversed_wm_data/MNIST_10/'):
            adv_mi_dataset = wmDataset(torch.load(root + files[0]), transform=datatransform)
            for file in files[1:]:
                adv_mi_dataset += wmDataset(torch.load(root + file), transform=datatransform)
        
    unlearning_dataset = Subset(clean_dataset, randint(0, len(clean_dataset),size=int(len(clean_dataset)/num_users)))  + adv_mi_dataset
    for n in range(num_adv):
        unlearning_dataset+=user_wm_dataset[n]

    return clean_dataset, unlearning_dataset, user_wm_dataset, test_dataset