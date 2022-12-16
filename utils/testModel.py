import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.others import *


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # for acGAN discriminator output two vector
            _, output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def comprehensive_test(model, device, clean_loader, v_wm_loader=None, a_wm_loader=None):
    print("Testing on clean test set")
    if v_wm_loader != None:
        print("Testing on victim's watermarking")
        test(model, device, v_wm_loader)
    if a_wm_loader != None:
        print("Testing on adversary's watermarking")
        test(model, device, a_wm_loader)


def comprehensive_user_test(model, device, clean_loader, user_wm_dataset):
    print("Testing on Users' datasset")
    test_list = []
    for user_id in range(len(user_wm_dataset)):
        user_wm_loader = DataLoader(user_wm_dataset[user_id], batch_size=1000, num_workers=2, shuffle=True)
        print('Testing on User ' + str(user_id) + ' watermark:')
        a = test(model, device, user_wm_loader)
        test_list.append(a)

    print("Testing on clean test set")
    a = test(model, device, clean_loader)
    test_list.append(a)
    return test_list


def testG(G, device, resul_dir):
    latent_size = 128
    
    nbImageToGenerate = 8*8
    for i in range(50):
        latent_value = torch.randn((nbImageToGenerate, latent_size)).to(device)
        gen_labels = torch.LongTensor(np.full(nbImageToGenerate , i)).to(device)
        fake_images = G(latent_value , gen_labels) #Generate a fake image
        save_image(fake_images), os.path.join(resul_dir, 'GeneratedSample-{}.png'.format)
    
def inversed_wm(G, device, resul_dir, target):
    latent_size = 128
    nb_classes = 10
    
    nbImageToGenerate = 8*8
    for i in range(50):
        #latent_value = torch.randn((nbImageToGenerate, latent_size)).to(device)
        latent_value = torch.normal(0,10,(nbImageToGenerate, latent_size)).to(device)
        gen_labels = torch.LongTensor(np.random.randint(0, nb_classes, 64)).to(device)
        index = np.where(((gen_labels).cpu()==target))
        index = np.array(index).reshape(-1)
        fake_images = G(latent_value , gen_labels) #Generate a fake image
        #fake_images = fake_images.view(-1,1,28,28)
        for idx in index:
            save_image(denorm(fake_images[idx]), os.path.join(resul_dir, 'GeneratedSample-{}-{}.png'.format(i,idx)))
    
