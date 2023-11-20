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
from models import Generator, ResNet18
from utils import one_hot, test_gen_backdoor, test, test_backdoor

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False, default='gtsrb',
                    choices=['mnist','gtsrb'])
parser.add_argument('-clean_budget', type=int, default=2000) # by defaut :  we assume 2000 clean samples for defensive purpose
parser.add_argument('-threshold', type=int, default=10)
parser.add_argument('-train_gen_epoch', type=int, default=30)
parser.add_argument('-gamma2', type=float, default=0.2)
parser.add_argument('-patch_rate', type=float, default=0.15)
args = parser.parse_args()

device='cuda' # for GPU only

# Initialize Datasets
# train_set/test_set: The train/test split of the dataset.
# source_class/target_class: A backdoor attack makes the model classify trigger-patched images from {source_class} to {target_class}.
# source_loader: Loader of samples from source_class to be patched.

if args.dataset=='mnist':
    source_class=1
    target_class=0
    data_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
        ])
    train_set=MNIST(root='/home/data',train=True,transform=data_transform)
    test_set=MNIST(root='/home/data',train=False,transform=data_transform)
    source_set=[]
    for img,label in train_set:
        if label==source_class:
            source_set.append((img,label))
    source_loader=DataLoader(source_set,batch_size=len(source_set),shuffle=False)
elif args.dataset=='gtsrb':
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

# we assume the defender can get {clean_budget} clean samples.
indices=np.random.choice(len(train_set), args.clean_budget, replace=False)
dataset=Subset(train_set,indices)
dataloader=DataLoader(dataset, batch_size=32, shuffle=True)

# Load Backdoored Model
model=ResNet18(num_classes=10 if args.dataset=='mnist' else 43)
model.load_state_dict(torch.load(f'backdoor_models/{args.dataset}-badnet.pt'))
model.eval()
model=model.to(device)

# Trigger Generation
print('---Trigger Generation---')

gen=Generator().to(device) # generator
optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4)

NLLLoss=nn.NLLLoss()
MSELoss=nn.MSELoss()
threshold=args.threshold

for epoch in range(args.train_gen_epoch):
    gen.train()
    Loss_sum=0
    L_trigger_sum=0
    L_pert_sum=0
    count_sum=0
    for i,(img,ori_label) in enumerate(dataloader):
        label=torch.ones_like(ori_label)*target_class
        one_hot_label=one_hot(label).to(device)
        img,label=img.to(device),label.to(device)
        noise=torch.randn((img.shape[0],100)).to(device)
        G_out=gen(one_hot_label,noise)
        D_out=model(img+G_out) # model as discriminator
        L_trigger=NLLLoss(D_out,label)
        G_out_norm=torch.norm(G_out, p=1)/img.shape[0] - threshold
        L_pert=torch.max(torch.zeros_like(G_out_norm), G_out_norm)
        Loss = L_trigger + args.gamma2*L_pert

        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        Loss_sum+=Loss.item()
        L_trigger_sum+=L_trigger.item()
        L_pert_sum+=L_pert.item()
        count_sum+=1
    bdacc=test_gen_backdoor(gen,model,source_loader,target_class,device) # test the ASR of the generated trigger
    print(f'Epoch-{epoch}: Loss={round(Loss_sum/count_sum,3)}, L_trigger={round(L_trigger_sum/count_sum,3)}, L_pert={round(L_pert_sum/count_sum,3)}, ASR={round(bdacc*100,2)}%')

# Model Patching
print('---Model Patching---')
gen.eval()
label=torch.ones((int(args.patch_rate*args.clean_budget)),dtype=torch.int64)*target_class
one_hot_label=one_hot(label).to(device)
noise=torch.randn((int(args.patch_rate*args.clean_budget),100)).to(device)
G_out=gen(one_hot_label,noise).detach().cpu()

patched_dataset=[]
for i,(img,label) in enumerate(dataset):
    if i<int(args.patch_rate*args.clean_budget):
        img=img+G_out[i]
    patched_dataset.append((img,label))

patched_loader=DataLoader(patched_dataset,batch_size=128,shuffle=True)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=1e-4)
criterion=nn.CrossEntropyLoss()

# load real backdoor trigger
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

# test Clean Acc and ASR before cleanse
clean_acc=test(model,test_set,device)
bd_acc=test_backdoor(model,patched_source_loader,device)
print(f'Before Cleanse. Clean Acc:{round(clean_acc*100,2)}%, ASR:{round(bd_acc*100,2)}%')
before_asr=bd_acc

# cleanse for 10 epochs
for epoch in range(10):
    model.train()
    for i,(img,label) in enumerate(patched_loader):
        img,label=img.to(device),label.to(device)
        out=model(img)
        loss=criterion(out,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    clean_acc=test(model,test_set,device)
    bd_acc=test_backdoor(model,patched_source_loader,device)
    print(f'Epoch-{epoch}. Clean Acc:{round(clean_acc*100,2)}%, ASR:{round(bd_acc*100,2)}%')
# test Clean Acc and ASR after cleanse
print(f'After Cleanse. Clean Acc:{round(clean_acc*100,2)}%, ASR:{round(bd_acc*100,2)}%, Backdoor(Watermark) Removal Rate:{round((before_asr-bd_acc)*100,2)}%.')