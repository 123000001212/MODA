import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models import alexnet, resnet18


class AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes, norm_type='bn', pretrained=False, imagenet=False):
        super(AlexNet, self).__init__()

        params = []

        if num_classes == 1000 or imagenet:  # imagenet1000
            if pretrained:
                norm_type = 'none'
            self.features = nn.Sequential(
                ConvBlock(3, 64, 11, 4, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ConvBlock(192, 384, 3, 1, 1, bn=norm_type),
                ConvBlock(384, 256, 3, 1, 1, bn=norm_type),
                ConvBlock(256, 256, 3, 1, 1, bn=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.AdaptiveAvgPool2d((6, 6))
            )

            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

            for layer in self.features:
                if isinstance(layer, ConvBlock):
                    params.append(layer.conv.weight)
                    params.append(layer.conv.bias)

            for layer in self.classifier:
                if isinstance(layer, nn.Linear):
                    params.append(layer.weight)
                    params.append(layer.bias)

            if pretrained:
                self._load_pretrained_from_torch(params)
        else:
            self.features = nn.Sequential(
                ConvBlock(in_channels, 64, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
                ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
                ConvBlock(192, 384, bn=norm_type),
                ConvBlock(384, 256, bn=norm_type),
                ConvBlock(256, 256, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
            )
            self.l_gan_logit = nn.Linear(4 * 4 * 256, 1) 
            self.classifier = nn.Linear(4 * 4 * 256, num_classes)

    def _load_pretrained_from_torch(self, params):
        # load a pretrained alexnet from torchvision
        torchmodel = alexnet(True)
        torchparams = []
        for layer in torchmodel.features:
            if isinstance(layer, nn.Conv2d):
                torchparams.append(layer.weight)
                torchparams.append(layer.bias)

        for layer in torchmodel.classifier:
            if isinstance(layer, nn.Linear):
                torchparams.append(layer.weight)
                torchparams.append(layer.bias)

        for torchparam, param in zip(torchparams, params):
            assert torchparam.size() == param.size(), 'size not match'
            param.data.copy_(torchparam.data)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        aux = self.classifier(x)
        gan = self.l_gan_logit(x)
        return torch.sigmoid(gan).squeeze(1), aux


class ConvBlock(nn.Module):
    def __init__(self, i, o, ks=3, s=1, pd=1, bn='bn', relu=True):
        super().__init__()

        self.conv = nn.Conv2d(i, o, ks, s, pd, bias=bn == 'none')

        if bn == 'bn':
            self.bn = nn.BatchNorm2d(o)
        elif bn == 'gn':
            self.bn = nn.GroupNorm(o // 16, o)
        elif bn == 'in':
            self.bn = nn.InstanceNorm2d(o)
        else:
            self.bn = None

        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_type='bn'):
        super(BasicBlock, self).__init__()

        self.convbnrelu_1 = ConvBlock(in_planes, planes, 3, stride, 1, bn=norm_type, relu=True)
        self.convbn_2 = ConvBlock(planes, planes, 3, 1, 1, bn=norm_type, relu=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = ConvBlock(in_planes, self.expansion * planes,
                                      1, stride, 0, bn=norm_type, relu=True)

    def forward(self, x):
        out = self.convbnrelu_1(x)
        out = self.convbn_2(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_type='bn'):
        super(Bottleneck, self).__init__()

        self.convbnrelu_1 = ConvBlock(in_planes, planes, 1, 1, 0, bn=norm_type, relu=True)
        self.convbnrelu_2 = ConvBlock(planes, planes, 3, stride, 1, bn=norm_type, relu=True)
        self.convbn_3 = ConvBlock(planes, self.expansion * planes, 1, 1, 0, bn=norm_type, relu=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = ConvBlock(in_planes, self.expansion * planes, 1, stride, 0, bn=norm_type, relu=False)

    def forward(self, x):
        out = self.convbnrelu_1(x)
        out = self.convbnrelu_2(out)
        out = self.convbn_3(out) + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_type='bn', pretrained=False, imagenet=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        self.norm_type = norm_type

        if num_classes == 1000 or imagenet:
            self.convbnrelu_1 = nn.Sequential(
                ConvBlock(3, 64, 7, 2, 3, bn=norm_type, relu=True),  # 112
                nn.MaxPool2d(3, 2, 1),  # 56
            )
        else:
            self.convbnrelu_1 = ConvBlock(3, 64, 3, 1, 1, bn=norm_type, relu=True)  # 32
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # 32/ 56
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 16/ 28
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 8/ 14
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 4/ 7
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.l_gan_logit = nn.Linear(512 * block.expansion, 1) 

        if num_classes == 1000 and pretrained:
            assert sum(num_blocks) == 8, 'only implemented for resnet18'
            layers = [self.convbnrelu_1[0].conv, self.convbnrelu_1[0].bn]
            for blocklayers in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for blocklayer in blocklayers:
                    b1 = blocklayer.convbnrelu_1
                    b2 = blocklayer.convbn_2
                    b3 = blocklayer.shortcut
                    layers += [b1.conv, b1.bn, b2.conv, b2.bn]
                    if not isinstance(b3, nn.Sequential):
                        layers += [b3.conv, b3.bn]
            layers += [self.linear]

            self._load_pretrained_from_torch(layers)

    def _load_pretrained_from_torch(self, layers):
        # load a pretrained alexnet from torchvision
        torchmodel = resnet18(True)
        torchlayers = [torchmodel.conv1, torchmodel.bn1]
        for torchblocklayers in [torchmodel.layer1, torchmodel.layer2, torchmodel.layer3, torchmodel.layer4]:
            for blocklayer in torchblocklayers:
                torchlayers += [blocklayer.conv1, blocklayer.bn1, blocklayer.conv2, blocklayer.bn2]
                if blocklayer.downsample is not None:
                    torchlayers += [blocklayer.downsample[0], blocklayer.downsample[1]]

        for torchlayer, layer in zip(torchlayers, layers):
            assert torchlayer.weight.size() == layer.weight.size(), 'must be same'
            layer.load_state_dict(torchlayer.state_dict())

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.convbnrelu_1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        aux = self.linear(out)
        gan = self.l_gan_logit(out)

        return torch.sigmoid(gan).squeeze(1), aux


def ResNet9(**model_kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], **model_kwargs)


def ResNet18(**model_kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **model_kwargs)


def ResNet34(**model_kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **model_kwargs)


def ResNet50(**model_kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **model_kwargs)


def ResNet101(**model_kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **model_kwargs)


def ResNet152(**model_kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **model_kwargs)
