import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST, GTSRB
import numpy as np
import argparse
import os
from PIL import Image
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False, default='gtsrb',
                    choices=['mnist','gtsrb'])
parser.add_argument('-clean_budget', type=int, default=2000)
parser.add_argument('-threshold', type=int, default=10)
parser.add_argument('-train_gen_epoch', type=int, default=30)
parser.add_argument('-gamma2', type=float, default=0.2)
parser.add_argument('-patch_rate', type=float, default=0.15)
# by defaut :  we assume 2000 clean samples for defensive purpose
args = parser.parse_args()

device='cuda'

import torch
from torchvision.utils import save_image
from torch.utils.data.dataloader import DataLoader
from PIL import Image

def one_hot(x, class_count=10):
    return torch.eye(class_count)[x, :]

def test_gen_backdoor(gen,model,source_loader,target_label,device):
    gen.eval()
    total_correct=0
    total_count=0
    with torch.no_grad():
        for i,(img,ori_label) in enumerate(source_loader):
            label=torch.ones_like(ori_label)*target_label
            one_hot_label=one_hot(label).to(device)
            img,label=img.to(device),label.to(device)
            noise=torch.randn((img.shape[0],100)).to(device)
            G_out=gen(one_hot_label,noise)
            D_out=model(img+G_out)
            pred = D_out.data.max(1)[1]
            total_correct += pred.eq(label.data.view_as(pred)).sum()
            total_count += img.shape[0]
    acc=total_correct/total_count
    return acc.item()

def test(model,test_set,device):
    model.eval()
    total_correct=0
    total_count=0
    test_loader=DataLoader(test_set,batch_size=1000,shuffle=False)
    with torch.no_grad():
        for i,(img,label) in enumerate(test_loader):
            img,label=img.to(device),label.to(device)
            out=model(img)
            pred = out.data.max(1)[1]
            total_correct += pred.eq(label.data.view_as(pred)).sum()
            total_count += img.shape[0]
    acc=total_correct/total_count
    return acc.item()

def test_backdoor(model,patched_source_loader,device):
    model.eval()
    total_correct=0
    total_count=0
    with torch.no_grad():
        for i,(img,label) in enumerate(patched_source_loader):
            img,label=img.to(device),label.to(device)
            out=model(img)
            pred = out.data.max(1)[1]
            total_correct += pred.eq(label.data.view_as(pred)).sum()
            total_count += img.shape[0]
    acc=total_correct/total_count
    return acc.item()

# ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_hidden=False, return_activation=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        activation1 = out
        out = self.layer2(out)
        activation2 = out
        out = self.layer3(out)
        activation3 = out
        out = self.layer4(out)
        out = self.avgpool(out)
        hidden = out.view(out.size(0), -1)
        out = self.linear(hidden)

        if return_hidden:
            return out, hidden
        elif return_activation:  # for NAD
            return out, activation1, activation2, activation3
        else:
            return out

    # for FeatureRE
    def from_input_to_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def from_features_to_output(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def get_layer(self, x, layer_output):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        activation1 = out
        out = self.layer2(out)
        activation2 = out
        out = self.layer3(out)
        activation3 = out
        out = self.layer4(out)
        out = self.avgpool(out)
        if layer_output == 'avgpool':
            return out
        else:
            raise NotImplementedError("`layer_output` must be 'avgpool'!")
        hidden = out.view(out.size(0), -1)
        out = self.linear(hidden)

    def freeze_feature(self):
        for name, para in self.named_parameters():
            if name.count('linear') == 0:  # non-linear layer
                para.requires_grad = False

    def unfreeze_feature(self):
        for name, para in self.named_parameters():
            para.requires_grad = True

    def freeze_fc(self):
        for name, para in self.named_parameters():
            if name.count('linear') > 0:  # linear layer
                para.requires_grad = False

    def unfreeze_fc(self):
        for name, para in self.named_parameters():
            if name.count('linear') > 0:  # linear layer
                para.requires_grad = True

    def freeze_before_last_block(self):
        for name, para in self.named_parameters():
            para.requires_grad = False

        self.linear.weight.requires_grad = True
        self.linear.bias.requires_grad = True

        last_block = self.layer3[-1]
        for name, para in last_block.named_parameters():
            para.requires_grad = True

    def unfreeze(self):
        for name, para in self.named_parameters():
            para.requires_grad = True


class ResNet_narrow(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_narrow, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 48, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_hidden=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        hidden = out.view(out.size(0), -1)
        out = self.linear(hidden)

        if return_hidden:
            return out, hidden
        else:
            return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet18_narrow(num_classes=10):
    return ResNet_narrow(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


""" gen=Generator()
c=torch.tensor([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]])
noise=torch.randn((2,100))
pic=gen(c,noise)
print(pic.shape)
a=1 """
if args.dataset=='gtsrb':
    source_class=1
    target_class=2
    data_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
    train_set = GTSRB(root='/home/data', split = 'train', transform = data_transform, download=True)
    test_set = GTSRB(root='/home/data', split = 'test', transform = data_transform, download=True)
    source_set=[]
    for img,label in train_set:
        if label==source_class:
            source_set.append((img,label))
    source_loader=DataLoader(source_set,batch_size=len(source_set),shuffle=False)
else:
    raise(NotImplementedError('Dataset not supported.'))

indices=np.random.choice(len(train_set), args.clean_budget, replace=False)
dataset=Subset(train_set,indices)
dataloader=DataLoader(dataset, batch_size=32, shuffle=True)

model=ResNet18(num_classes=10 if args.dataset=='mnist' else 43)
model.load_state_dict(torch.load('backdoor_models/gtsrb-badnet.pt'))
model.eval()
model=model.to(device)

# real backdoor trigger
totensor=transforms.ToTensor()
trigger = Image.open('triggers/badnet_patch_32.png').convert("RGB")
trigger = totensor(trigger)
trigger_mask = Image.open('triggers/mask_badnet_patch_32.png').convert("RGB")
trigger_mask = totensor(trigger_mask)
trigger_mask = trigger_mask[0]
patched_source_set=[]
for img,label1 in source_set:
    img=img.clone()
    img=img+trigger_mask*(trigger-img)
    patched_source_set.append((img,target_class))
patched_source_loader=DataLoader(patched_source_set,batch_size=1000,shuffle=False)

clean_acc=test(model,test_set,device)
bd_acc=test_backdoor(model,patched_source_loader,device)
print(f'Before Cleanse. Clean Acc:{round(clean_acc*100,2)}%, ASR:{round(bd_acc*100,2)}%')

