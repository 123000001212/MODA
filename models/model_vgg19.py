import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv4=nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3,padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3,padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.conv11= nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.conv12= nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.pool4=nn.MaxPool2d(kernel_size=2,stride=2,padding=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.pool5=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(7*7*512,4096)
        self.fc2=nn.Linear(4096,4096)
        self.fc3=nn.Linear(4096,10)

        self.relu=nn.ReLU()
        self.softmax=nn.Softmax(dim=1)
        print("success")
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.pool3(x)
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.pool4(x)
        x = self.relu(self.conv13(x))
        x = self.relu(self.conv14(x))
        x = self.relu(self.conv15(x))
        x = self.relu(self.conv16(x))
        x = self.pool5(x)
        # x = x.view(x.size()[0], -1)
        x = x.view(-1,7*7*512)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        output = self.softmax(self.fc3(x))
        return output


class VGG_pretrained(nn.Module):
    def __init__(self):
        super(VGG_pretrained, self).__init__()
        self.features = torchvision.models.vgg19(pretrained=False).features
        '''
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 10, bias=False)
        )
        '''
        self.l_gan_logit = nn.Linear(512, 1) 
        self.classifier = nn.Linear(512, 10) 

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # method 2 kaiming
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        aux = self.classifier(x)
        gan = self.l_gan_logit(x)
        return torch.sigmoid(gan).squeeze(1), aux