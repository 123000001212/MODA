import torch
import torch.nn as nn
import torch.nn.functional as F


class ResamplingConv(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, padding=None, resample=None, bias=True):
        """
        Combines resampling (up/down) with convolution.
        If resample is "up", then nn.Upsample(x2) is applied before the conv
        If resample is "down", then nn.MaxPool2D(x2) is applied after the conv
        Padding is automatically set to preserve spatial shapes of input.
        If kernel is 0, no conv is applied and the module is reduced to up/down sampling (if any)
        Resample=None and kernel=0  ==>  nothing happens
        """
        super(ResamplingConv, self).__init__()
        assert(kernel in (0, 1, 3, 5, 7))
        padding = (kernel - 1) // 2 if padding is None else padding
        if kernel == 0:
            self.conv = None
        else:
            self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel, padding=padding, bias=bias)
        self.resample = resample
        if resample == "down":
            self.resampler = torch.nn.AvgPool2d(2)

    def forward(self, x):
        if self.resample == "up":
            x = F.interpolate(x, scale_factor=2)
        if self.conv is not None:
            x = self.conv(x)
        if self.resample == "down":
            x = self.resampler(x)
        return x


class ResBlock(torch.nn.Module):
    def __init__(self, inplanes, planes, kernel=3, resample=None, bias=True, batnorm=True):
        """
        Residual block with two convs and optional resample.
        If resample == "up", the first conv is upsampling by 2, residual is upsampled too
        If resample == "down", the last conv is downsampling by 2, residual is downsampled too
        If resample == None, no resampling anywhere
        1x1 conv is applied to residual only if inplanes != planes and resample is None.
        """
        super(ResBlock, self).__init__()
        self.conv1 = ResamplingConv(inplanes, planes, kernel,
                                    resample=resample if resample != "down" else None,
                                    bias=bias)
        self.bn1 = torch.nn.BatchNorm2d(planes) if batnorm else None
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = ResamplingConv(planes, planes, kernel,
                                    resample=resample if resample != "up" else None,
                                    bias=bias)
        self.bn2 = torch.nn.BatchNorm2d(planes) if batnorm else None
        self.resample = resample
        self.shortcut = ResamplingConv(inplanes, planes, 0 if (inplanes == planes and resample is None) else 1,
                                       resample=resample,
                                       bias=bias)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out) if self.bn1 is not None else out
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out) if self.bn2 is not None else out

        residual = self.shortcut(residual)

        out += residual
        out = self.relu(out)

        return out

# endregion


class Generator(torch.nn.Module):
    def __init__(self, z_dim, dim_g, nb_classes, **kw):
        super(Generator, self).__init__(**kw)
        self.label_embedding = nn.Embedding(nb_classes, 128)
        self.conv1 = torch.nn.ConvTranspose2d(z_dim, dim_g, 4)
        self.bn1 = torch.nn.BatchNorm2d(dim_g)
        self.res1 = ResBlock(dim_g, dim_g, 3, resample='up')
        self.res2 = ResBlock(dim_g, dim_g, 3, resample='up')
        self.res3 = ResBlock(dim_g, dim_g, 3, resample='up')
        self.conv2 = torch.nn.Conv2d(dim_g, 3, 3, padding=1)

    def forward(self, x, cl):
        x = torch.mul(self.label_embedding(cl), x)
        x = x.unsqueeze(2).unsqueeze(3)
        x =self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        return x


# endregion
class Discriminator(torch.nn.Module):
    def __init__(self, dim_d, num_classes, **kw):
        super(Discriminator, self).__init__(**kw)
        self.res1 = ResBlock(3, dim_d, 3, resample='down', batnorm=False)
        self.res2 = ResBlock(dim_d, dim_d, 3, resample='down', batnorm=False)
        self.res3 = ResBlock(dim_d, dim_d, 3, resample=None, batnorm=False)
        self.res4 = ResBlock(dim_d, dim_d, 3, resample=None, batnorm=False)
        self.gan_linear = nn.Linear(dim_d, 1)
        self.aux_linear = nn.Linear(dim_d, num_classes)
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = x.mean(3).mean(2)
        c = self.aux_linear(x)
        s = self.gan_linear(x)
        s = torch.sigmoid(s)
        return s.squeeze(1), c.squeeze(1)


# region oldmodel
class Normalize(torch.nn.Module):
    def __init__(self, dim, **kw):
        super(Normalize, self).__init__(**kw)
        self.bn = torch.nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(x)


class ConvMeanPool(torch.nn.Module):
    def __init__(self, indim, outdim, filter_size, biases=True, **kw):
        super(ConvMeanPool, self).__init__(**kw)
        assert(filter_size % 2 == 1)
        padding = filter_size // 2
        self.conv = torch.nn.Conv2d(indim, outdim, kernel_size=filter_size, padding=padding, bias=biases)
        self.pool = torch.nn.AvgPool2d(2)

    def forward(self, x):
        y = self.conv(x)
        y = self.pool(y)
        return y


class MeanPoolConv(ConvMeanPool):
    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y)
        return y


class UpsampleConv(torch.nn.Module):
    def __init__(self, indim, outdim, filter_size, biases=True, **kw):
        super(UpsampleConv, self).__init__(**kw)
        assert(filter_size % 2 == 1)
        padding = filter_size // 2
        self.conv = torch.nn.Conv2d(indim, outdim, kernel_size=filter_size, padding=padding, bias=biases)


    def forward(self, x):
        y = F.interpolate(x, scale_factor=2)
        y = self.conv(y)
        return y


class ResidualBlock(torch.nn.Module):
    def __init__(self, indim, outdim, filter_size, resample=None, use_bn=False, **kw):
        super(ResidualBlock, self).__init__(**kw)
        assert(filter_size % 2 == 1)
        padding = filter_size // 2
        bn2dim = outdim
        if resample == "down":
            self.conv1 = torch.nn.Conv2d(indim, indim, kernel_size=filter_size, padding=padding, bias=True)
            self.conv2 = ConvMeanPool(indim, outdim, filter_size=filter_size)
            self.conv_shortcut = ConvMeanPool
            bn2dim = indim
        elif resample == "up":
            self.conv1 = UpsampleConv(indim, outdim, filter_size=filter_size)
            self.conv2 = torch.nn.Conv2d(outdim, outdim, kernel_size=filter_size, padding=padding, bias=True)
            self.conv_shortcut = UpsampleConv
        else:   # None
            assert(resample is None)
            self.conv1 = torch.nn.Conv2d(indim, outdim, kernel_size=filter_size, padding=padding, bias=True)
            self.conv2 = torch.nn.Conv2d(outdim, outdim, kernel_size=filter_size, padding=padding, bias=True)
            self.conv_shortcut = torch.nn.Conv2d
        if use_bn:
            self.bn1 = Normalize(indim)
            self.bn2 = Normalize(bn2dim)
        else:
            self.bn1, self.bn2 = None, None

        self.nonlin = torch.nn.ReLU()

        if indim == outdim and resample == None:
            self.conv_shortcut = None
        else:
            self.conv_shortcut = self.conv_shortcut(indim, outdim, filter_size=1)       # bias is True by default, padding is 0 by default

    def forward(self, x):
        if self.conv_shortcut is None:
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)
        y = self.bn1(x) if self.bn1 is not None else x
        y = self.nonlin(y)
        y = self.conv1(y)
        y = self.bn2(y) if self.bn2 is not None else y
        y = self.nonlin(y)
        y = self.conv2(y)

        return y + shortcut


class OptimizedResBlockDisc1(torch.nn.Module):
    def __init__(self, dim, **kw):
        super(OptimizedResBlockDisc1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, dim, kernel_size=3, padding=1, bias=True)
        self.conv2 = ConvMeanPool(dim, dim, filter_size=3, biases=True)
        self.conv_shortcut = MeanPoolConv(3, dim, filter_size=1, biases=True)
        self.nonlin = torch.nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.nonlin(y)
        y = self.conv2(y)
        shortcut = self.conv_shortcut(x)
        return y + shortcut


class OldGenerator(torch.nn.Module):
    def __init__(self, z_dim, dim_g, nb_classes, use_bn=True, **kw):
        super(OldGenerator, self).__init__(**kw)
        self.dim_g = dim_g

        self.label_embedding = nn.Embedding(nb_classes, 128)
        self.linear = nn.Linear(z_dim, 4*4*dim_g)
        self.res1 = ResidualBlock(dim_g, dim_g, 3, resample="up", use_bn=use_bn)
        self.res2 = ResidualBlock(dim_g, dim_g, 3, resample="up", use_bn=use_bn)
        self.res3 = ResidualBlock(dim_g, dim_g, 3, resample="up", use_bn=use_bn)
        self.normal = Normalize(dim_g)
        self.conv = torch.nn.Conv2d(dim_g, 3, kernel_size=3, padding=1)

    def forward(self, x, cl):
        x = torch.mul(self.label_embedding(cl), x)
        x = self.linear(x)
        x = x.view(x.size(0), self.dim_g, 4, 4)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.normal(x)
        x = F.relu(x)
        x = self.conv(x)
        x = torch.tanh(x)
        return x


class OldDiscriminator(torch.nn.Module):
    def __init__(self, dim_d, num_classes, use_bn=False, **kw):
        super(OldDiscriminator, self).__init__(**kw)
        self.optimRes = OptimizedResBlockDisc1(dim_d)
        self.res1 = ResidualBlock(dim_d, dim_d, 3, resample="down", use_bn=use_bn)
        self.res2 = ResidualBlock(dim_d, dim_d, 3, resample=None, use_bn=use_bn)
        self.res3 = ResidualBlock(dim_d, dim_d, 3, resample=None, use_bn=use_bn)
        self.gan_linear = nn.Linear(dim_d, 1)
        self.aux_linear = nn.Linear(dim_d, num_classes)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.optimRes(x)
        x = self.res1(x)
        x = self.dropout(x)
        x = self.res2(x)
        x = self.dropout(x)
        x = self.res3(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = x.mean(3).mean(2)
        c = self.aux_linear(x)
        s = self.gan_linear(x)
        s = torch.sigmoid(s)
        return s.view(-1, 1), c.squeeze(1)
# endregion


class UnquantizeTransform(object):
    def __init__(self, levels=256, range=(-1., 1.)):
        super(UnquantizeTransform, self).__init__()
        self.rand_range = (range[1] - range[0]) * 1. / (1. * levels)

    def __call__(self, x):
        rand = torch.rand_like(x) * self.rand_range
        x = x + rand
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(rand_range={0})'.format(self.rand_range)