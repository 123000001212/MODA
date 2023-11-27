
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from model_init import *
from dataset_init import *
from utils.others import *
from utils.testModel import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current Device " , device)

batch_size = 64
num_epochs = 1000
num_users = 6
nb_classes = 10
dataset = 'MNIST'

clean_dataset, unlearning_dataset, user_wm_dataset, test_dataset = unlearning_dataset_init(dataset, 6)

train_loader = DataLoader(unlearning_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

D, G = model_init(dataset, device)
D.load_state_dict(torch.load('./D-CL.pth'))
optimizerD = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), 0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(G.parameters(), 0.0002, betas=(0.5, 0.999))
criterion_adv = nn.BCELoss()
criterion_aux = nn.CrossEntropyLoss() 

total_step = len(train_loader)
comprehensive_user_test(D, device, test_loader, user_wm_dataset)
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
    comprehensive_user_test(D, device, test_loader, user_wm_dataset)
