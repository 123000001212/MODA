from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
# import utils.torchlib.layers

class NoOp(nn.Module):

    def __init__(self, *args, **keyword_args):
        super(NoOp, self).__init__()

    def forward(self, x):
        return x
    
class Reshape(nn.Module):

    def __init__(self, *new_shape):
        super(Reshape, self).__init__()
        self._new_shape = new_shape

    def forward(self, x):
        new_shape = (x.size(i) if self._new_shape[i] == 0 else self._new_shape[i] for i in range(len(self._new_shape)))
        return x.view(*new_shape)
    
def identity(x, *args, **keyword_args):
    return x

def _get_norm_fn_2d(norm):  # 2d
    if norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == 'instance_norm':
        return nn.InstanceNorm2d
    elif norm == 'none':
        return NoOp
    else:
        raise NotImplementedError
        
def _get_weight_norm_fn(weight_norm):
    if weight_norm == 'spectral_norm':
        return torch.nn.utils.spectral_norm
    elif weight_norm == 'weight_norm':
        return torch.nn.utils.weight_norm
    elif weight_norm == 'none':
        return identity
    else:
        return NotImplementedError

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

        self.layer6 = nn.Sequential(nn.ConvTranspose2d(32,3,1,1,0,bias = False))

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

class Discriminator(nn.Module):

    def __init__(self, x_dim, c_dim, dim=96, norm='none', weight_norm='spectral_norm'):
        super(Discriminator, self).__init__()

        norm_fn = _get_norm_fn_2d(norm)
        weight_norm_fn = _get_weight_norm_fn(weight_norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                weight_norm_fn(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)),
                norm_fn(out_dim),
                nn.LeakyReLU(0.2)
            )

        self.ls = nn.Sequential(  # (N, x_dim, 32, 32)
            conv_norm_lrelu(x_dim, dim),
            conv_norm_lrelu(dim, dim),
            conv_norm_lrelu(dim, dim, stride=2),  # (N, dim , 16, 16)

            conv_norm_lrelu(dim, dim * 2),
            conv_norm_lrelu(dim * 2, dim * 2),
            conv_norm_lrelu(dim * 2, dim * 2, stride=2),  # (N, dim*2, 8, 8)

            conv_norm_lrelu(dim * 2, dim * 2, kernel_size=3, stride=1, padding=0),
            conv_norm_lrelu(dim * 2, dim * 2, kernel_size=1, stride=1, padding=0),
            conv_norm_lrelu(dim * 2, dim * 2, kernel_size=1, stride=1, padding=0),  # (N, dim*2, 6, 6)

            nn.AvgPool2d(kernel_size=6),  # (N, dim*2, 1, 1)
            Reshape(-1, dim * 2),  # (N, dim*2)
        )

        self.l_gan_logit = weight_norm_fn(nn.Linear(dim * 2, 1))  # (N, 1)
        self.l_c_logit = nn.Linear(dim * 2, c_dim)  # (N, c_dim)

    def forward(self, x):
        # x: (N, x_dim, 32, 32)
        feat = self.ls(x)
        gan_logit = self.l_gan_logit(feat)
        l_c_logit = self.l_c_logit(feat)
        return torch.sigmoid(gan_logit).squeeze(1), l_c_logit

