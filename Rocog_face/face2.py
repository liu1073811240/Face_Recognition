import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F

# 创建arcsoftmax分类器，创建人脸特征提取器（使用预训练模型
class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.W = nn.Parameter(torch.randn((feature_num, cls_num)), requires_grad=True)
        self.func = nn.Softmax()

    def forward(self, x, s=1, m=0.2):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.W, dim=0)

        cosa = torch.matmul(x_norm, w_norm) / 10
        a = torch.acos(cosa)

        arcsoftmax = torch.exp(
            s * torch.cos(a + m) * 10) / (torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True) - torch.exp(
            s * cosa * 10) + torch.exp(s * torch.cos(a + m) * 10))

        return arcsoftmax

class FaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # 112*112
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 62*62
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 31*31

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 15*15
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 7*7

            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 3*3
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),


        )
        self.feature = nn.Linear(128*3*3, 128)  # 特征向量128维
        self.arc_softmax = Arcsoftmax(128, 27)

    def forward(self,x):
        # print(x.shape)  # torch.Size([100, 3, 112, 112])
        y = self.conv_layer(x)
        # print(y.shape)  # torch.Size([100, 32, 3, 3])
        y = torch.reshape(y, [-1, 128 * 3 * 3])
        feature = self.feature(y)
        # print(feature.shape)  # torch.Size([100, 512])
        # print(self.arc_softmax(feature).shape, 1, 1)  # torch.Size([100, 26])
        return feature, self.arc_softmax(feature,1,1)

    def encode(self,x):
        y = self.conv_layer(x)
        y = torch.reshape(y, [-1, 128 * 3 * 3])
        feature = self.feature(y)  # N,128
        # print(feature.shape)
        return feature

def compare(face1, face2):
    face1_norm = F.normalize(face1)
    face2_norm = F.normalize(face2)
    # print(face1_norm.shape)  # torch.Size([1, 512])
    # print(face2_norm.shape)  # torch.Size([1, 512])
    cosa = torch.matmul(face1_norm, face2_norm.T)
    return cosa

if __name__ == '__main__':
    a = torch.randn(100, 3, 112, 112)
    net = FaceNet()
    net(a)
    net.encode(a)
    # print(models.mobilenet_v2())


