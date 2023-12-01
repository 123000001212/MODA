# Unlearning victims' watermarks.

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_init import *
from dataset_init import *
from utils.others import *
from utils.testModel import *

parser = argparse.ArgumentParser(description='MODA')
parser.add_argument('--dataset', default='MNIST',help="datasets e.g. MNIST|SVHN|CIFAR10|GTSRB")
parser.add_argument('--batch_size', default=64,type=int, help='batch size for target data')
parser.add_argument('--num_epochs', default=10, type=float, help="number of epochs")
parser.add_argument('--num_users', default=3, type=float, help="number of users")

args = parser.parse_args()
# setups
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current Device " , device)
batch_size = args.batch_size
num_epochs = args.num_epochs
num_users = args.num_users
dataset = args.dataset

# dataset initialize
clean_dataset, unlearning_dataset, user_wm_dataset, test_dataset = unlearning_dataset_init(dataset, num_users)

# dataloader initialize
train_loader = DataLoader(unlearning_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# load model
D, G = model_init(dataset, device)
D.load_state_dict(torch.load('./checkpoints/D-CL.pth')) # model can be trained using train_ACGAN.ipynb

# optimizer/criterion initialize
optimizerD = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), 0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(G.parameters(), 0.0002, betas=(0.5, 0.999))
criterion_adv = nn.BCELoss()
criterion_aux = nn.CrossEntropyLoss() 

total_step = len(train_loader)
comprehensive_user_test(D, device, test_loader, user_wm_dataset) # test before unlearn

# unlearning
for epoch in range(num_epochs):
    for batch_idx, (images, target) in enumerate(train_loader):
        images, target = images.to(device), torch.LongTensor(target).to(device)
        predictR, predictRLabel = D(images) #image from the real dataset
        loss_real_aux = criterion_aux(predictRLabel, target)
        optimizerD.zero_grad()
        optimizerG.zero_grad()
        loss_real_aux.backward()
        optimizerD.step()
        real_score = predictR
    print("epoch",epoch)
    comprehensive_user_test(D, device, test_loader, user_wm_dataset) # test after every epoch
