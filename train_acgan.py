import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from numpy.random import randint
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from model_init import *
from dataset_init import *
from utils.others import *
from utils.testModel import *
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current Device " , device)

resul_dir = './results'
if not os.path.exists(resul_dir):
    os.makedirs(resul_dir)

batch_size = 128
num_epochs = 1000
num_users = 3
nb_classes = 10
dataset = 'MNIST'

#dataset initialize
clean_dataset, train_dataset, user_wm_dataset, test_dataset = dataset_init(dataset, num_users=num_users)

adv_mi_dataset = Subset(clean_dataset, randint(0, len(clean_dataset),size=300)) # 1800 300

for i in range(num_users-1):
    adv_mi_dataset += Subset(user_wm_dataset[i], randint(0, len(user_wm_dataset[i]),size=300))

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

adv_loader = DataLoader(adv_mi_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)

test_loader = DataLoader(test_dataset, batch_size=1000, num_workers=4, shuffle=False)

D, G = model_init(dataset, device) # model initialize
optimizerG = torch.optim.Adam(G.parameters(), 0.0002, betas=(0.5, 0.999))

optimizerD = torch.optim.SGD(filter(lambda p: p.requires_grad, D.parameters()),lr=0.01, momentum=0.9, weight_decay=0.0001)
criterion_adv = nn.BCELoss()
criterion_aux = nn.CrossEntropyLoss() 

#trian acgan
total_step = len(train_loader)
for epoch in range(num_epochs):
    
    for batch_idx, data in enumerate(zip(train_loader, cycle(adv_loader))):
        time2 = time.time()
        x, target = data[0]
        images = x.to(device)
        target = torch.LongTensor(target).to(device)
        # TRAIN D
        # On true data in collaborative learning
        froze_layer(D.l_gan_logit)
        predictR, predictRLabel = D(images) #image from the real dataset
        loss_real_aux = criterion_aux(predictRLabel, target)
        optimizerD.zero_grad()
        optimizerG.zero_grad()
        loss_real_aux.backward()
        optimizerD.step()

        if batch_idx ==0:
            time_normal = time.time()-time2
            print('The use of a normal batch is {}'.format(time_normal))
        real_score = predictR

        activate_layer(D.l_gan_logit)

        # On MI data
        time3 = time.time()
        x, target = data[1]
        images = x.to(device)
        target = torch.LongTensor(target).to(device)

        current_batchSize = images.size()[0]
        realLabel = torch.ones(current_batchSize).to(device)
        fakeLabel = torch.zeros(current_batchSize).to(device)

        predictR, predictRLabel = D(images)
        loss_real_aux = criterion_aux(predictRLabel, target)
        loss_real_adv = criterion_adv(predictR, realLabel)
        real_score = predictR

        # On fake data
        latent_value = torch.normal(0,10,(current_batchSize, 128)).to(device)
        gen_labels = torch.LongTensor(np.random.randint(0, nb_classes, current_batchSize)).to(device)
        fake_images = G(latent_value , gen_labels) #generate a fake image
        predictF, predictFLabel = D(fake_images)
        loss_fake_adv = criterion_adv(predictF ,  fakeLabel) # compare vs label =0 (D is supposed to "understand" that the image generated by G is fake)
        loss_fake_aux = criterion_aux(predictFLabel, gen_labels)
        fake_score = predictF

        lossD = loss_real_adv + loss_real_aux  +loss_fake_adv + loss_fake_aux

        optimizerD.zero_grad()
        optimizerG.zero_grad()
        lossD.backward()
        optimizerD.step()
        for i in range(6): 
        # TRAIN G
            latent_value = torch.normal(0,10,(current_batchSize, 128)).to(device)
            gen_labels = torch.LongTensor(np.random.randint(0, nb_classes, current_batchSize)).to(device)
            fake_images= G(latent_value, gen_labels) #Generate a fake image
            predictG, predictLabel = D(fake_images)
            lossG_adv = criterion_adv(predictG, realLabel) # Compare vs label = 1 (We want to optimize G to fool D, predictG must tend to 1)
            lossG_aux = criterion_aux(predictLabel, gen_labels)
            lossG = lossG_adv + lossG_aux
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
        if batch_idx == 0:
            time_acgan = time.time()-time3
            print('The time use of acgan is {}'.format(time_acgan))

        if (batch_idx+1) % 100 == 0:
            print("Epoch: "+str(epoch)+"/"+str(num_epochs)+ "  -- Batch:"+ str(batch_idx+1)+"/"+str(total_step))
            print("     GenLoss "+str(round(lossG.item(), 3))+ "  --  DiscLoss "+str(round(lossD.item(), 3)))
            print("     D(x): "+str(round(real_score.mean().item(), 3))+ "  -- D(G(z)):"+str(round(fake_score.mean().item(), 3)))



    with torch.no_grad():
        fake_images = fake_images.reshape(fake_images.size(0), 3, 32, 32)
        save_image(denorm(fake_images), os.path.join(resul_dir, 'fake_images-{}.png'.format(epoch+1)))
    if (epoch+1) == 1:
        save_image(images, os.path.join(resul_dir, 'real_images.png'),  normalize = True)
    
        
    test = comprehensive_user_test(D, device, test_loader, user_wm_dataset)
  
    
    torch.save(G.state_dict(), './checkpoints/G-CL.pth')
    torch.save(D.state_dict(), './checkpoints/D-CL.pth')