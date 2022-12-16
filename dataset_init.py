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

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index): 
        fn, label = self.imgs[index]  
        img = Image.open(fn).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img)

        return img, label  

    def __len__(self): 
        return len(self.imgs)

def dataset_init(dataset):
    if dataset == 'MNIST':
        dataTransform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5 ))])
        clean_dataset = datasets.MNIST(root='./data/', train=True,download=True,transform=dataTransform)
        test_dataset = datasets.MNIST(root='./data/', train=False,download=True,transform=dataTransform)
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
        clean_dataset = datasets.FashionMNIST('./dataset', train=True, transform=dataTransform)
        test_dataset = datasets.FashionMNIST('./dataset', train=False, transform=dataTransform)
    if dataset == 'GTSRB':
        dataTransform = transforms.Compose([
                                        transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
        train_dataset_root = '/home/data/GTSRB/GTSRB/Final_Training/Images'
        clean_dataset = datasets.ImageFolder(train_dataset_root, transform=dataTransform)
        test_dataset = MyDataset(txt='/home/data/GTSRB/GTSRB/Final_Test/test_dataset.txt', transform=dataTransform)
    if dataset == 'CIFAR10':
        dataTransform = transforms.Compose([
                            transforms.CenterCrop(32),
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
        clean_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=dataTransform)
        test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=dataTransform)
    user_wm_dataset,  train_dataset= [], copy.deepcopy(clean_dataset)

    for root, dirs, files in os.walk('./wm_data/' + dataset + '/'): 
        files.sort()
        for file in files:
            user_wm_dataset.append(wmDataset(torch.load(root + file)))
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
        train_dataset_root = '/home/data/GTSRB/GTSRB/Final_Training/Images'
        clean_dataset = datasets.ImageFolder(train_dataset_root, transform=dataTransform)
        test_dataset = MyDataset(txt='/home/data/GTSRB/GTSRB/Final_Test/test_dataset.txt', transform=dataTransform)
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
    for root, dirs, files in os.walk('./wm_data/' + dataset + '/'): 
        files.sort()
        for file in files:
            user_wm_dataset.append(wmDataset(torch.load(root + file)))
    if num_users==3:
        adv_mi_dataset = wmDataset(torch.load('./inversed_wm_data/' + dataset + '/unlearn_trigger.pth'), transform=dataTransform) + wmDataset(torch.load('/home/linshen/MODA/inversed_wm_data/' + dataset + '/unlearn_unrelated.pth'), transform=dataTransform)
    elif num_users==2:
        adv_mi_dataset = wmDataset(torch.load('./inversed_wm_data/' + dataset + '/unlearn_trigger.pth'), transform=dataTransform)
    else:
        for root, dirs, files in os.walk('./inversed_wm_data/MNIST/'):
            adv_mi_dataset = wmDataset(torch.load(root + files[0]), transform=datatransform)
            for file in files[1:]:
                adv_mi_dataset += wmDataset(torch.load(root + file), transform=datatransform)

        
    unlearning_dataset = Subset(clean_dataset, randint(0, len(clean_dataset),size=int(len(clean_dataset)/num_users))) + user_wm_dataset[-1] + adv_mi_dataset

    return clean_dataset, unlearning_dataset, user_wm_dataset, test_dataset

