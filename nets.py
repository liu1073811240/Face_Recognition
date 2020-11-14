import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch import sigmoid, relu


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1),  # conv1
            nn.BatchNorm2d(num_features=10),
            nn.PReLU(),  # PReLU1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
            nn.BatchNorm2d(num_features=16),
            nn.PReLU(),  # PReLU2
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
            nn.BatchNorm2d(num_features=32),
            nn.PReLU()  # PReLU3
        )

        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        # self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pre_layer(x)
        cond = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return cond, offset


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1, padding=1),  # conv1(24*24*28)
            nn.BatchNorm2d(num_features=28),
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1(11*11*28)
            nn.Conv2d(28, 48, kernel_size=3, stride=1),  # conv2(9*9*48)
            nn.BatchNorm2d(num_features=48),
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2(4*4*48)
            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # conv3(3*3*64)
            nn.BatchNorm2d(num_features=64),
            nn.PReLU()  # prelu3

        )
        self.conv4 = nn.Linear(64 * 3 * 3, 128)  # conv4
        self.prelu4 = nn.PReLU()  # prelu4
        # detection
        self.conv5_1 = nn.Linear(128, 1)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)

    def forward(self, x):
        # backend
        x = self.pre_layer(x)

        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)
        # detection
        label = sigmoid(self.conv5_1(x))
        offset = self.conv5_2(x)
        return label, offset


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # conv1 48*48
            nn.BatchNorm2d(num_features=32),
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1 23*23
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2 21*21
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2 10*10
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3 8*8
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(),  # prelu3
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool3 4*4
            nn.Conv2d(64, 128, kernel_size=2, stride=1),  # conv4 3*3
            nn.BatchNorm2d(num_features=128),
            nn.PReLU()  # prelu4
        )
        self.conv5 = nn.Linear(128 * 3 * 3, 256)  # conv5
        self.prelu5 = nn.PReLU()  # prelu5
        # detection
        self.conv6_1 = nn.Linear(256, 1)
        # bounding box regression
        self.conv6_2 = nn.Linear(256, 4)

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        # detection
        label = sigmoid(self.conv6_1(x))
        offset = self.conv6_2(x)
        return label, offset