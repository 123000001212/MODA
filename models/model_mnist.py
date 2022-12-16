import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
#Discriminator model
class Discriminator(nn.Module):
    def __init__(self , input_size, hidden_size):
         super(Discriminator, self).__init__()
         self.linear1 = nn.Linear(input_size , hidden_size)
         self.dropout = nn.Dropout(0.9)
         self.linear2 = nn.Linear(hidden_size , hidden_size)
         self.linear3 = nn.Linear(hidden_size, 1)
         self.linearAUx = nn.Linear(hidden_size, 10)
     #image and label
    def forward(self, x):
        x = x.view(x.shape[0],1,-1)
        x = F.leaky_relu(self.linear1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x), 0.2)
        x = self.dropout(x)
        output = torch.sigmoid((self.linear3(x)))
        aux = self.linearAUx(x)
        return output.view(output.shape[0],-1), F.log_softmax(aux.view(aux.shape[0],-1), dim=1)

# Generator Model
# class Generator(nn.Module):
#     def __init__(self , input_size, hidden_size, output_size):
#          super(Generator, self).__init__()
#          self.linear1 = nn.Linear(input_size, hidden_size)
#          self.linear2 = nn.Linear(hidden_size , hidden_size)
#          self.linear3 = nn.Linear(hidden_size, output_size)
#          self.label_embedding = nn.Embedding(10, input_size)
#     # x random  y labels
#     def forward(self, x, y):
#         x  = torch.mul(self.label_embedding(y), x)
#         x = F.relu(self.linear1(x))
#         x = F.relu(self.linear2(x))
#         x= self.linear3(x)
#         return torch.tanh(x) #Tied Sigmoid instead : did not work

class Generator(nn.Module):

    def __init__(self, latent_size , nb_filter, nb_classes):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(nb_classes, latent_size)
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(128,512,4,1,1,bias = False),
                                   nn.ReLU(True))

        #input 512*4*4
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(512,256,4,2,1,bias = False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True))
        #input 256*8*8
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1,bias = False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(True))
        #input 128*16*16
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1,bias = False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True))
        #input 64*32*32
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(64,32,4,2,1,bias = False),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(True))

        self.layer6 = nn.Sequential(nn.ConvTranspose2d(32,1,1,1,0,bias = False))

    def forward(self, input, cl):
        x = torch.mul(self.label_embedding(cl.long()), input)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return torch.tanh(x)
