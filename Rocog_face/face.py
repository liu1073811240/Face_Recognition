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

    def forward(self, x, s=1, m=0.2):  # s=64， m=0.5
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
        self.sub_net = nn.Sequential(
            models.mobilenet_v2(pretrained=True),
        )
        self.feature_net = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.1),
            nn.Linear(1000, 512, bias=False),  # 128, 256, 512
        )
        self.arc_softmax = Arcsoftmax(512, 27)

    def forward(self, x):
        # print(x.shape)  # torch.Size([100, 3, 28, 28])
        y = self.sub_net(x)
        # print(y.shape)  # torch.Size([100, 1000])
        feature = self.feature_net(y)
        # print(feature.shape)  # torch.Size([100, 512])
        # print(self.arc_softmax(feature, 1, 1).shape)  # torch.Size([100, 26])
        return feature, self.arc_softmax(feature, 1, 1)

    def encode(self, x):
        return self.feature_net(self.sub_net(x))


def compare(face1, face2):
    face1_norm = F.normalize(face1)
    face2_norm = F.normalize(face2)
    # print(face1_norm.shape)  # torch.Size([1, 512])
    # print(face2_norm.shape)  # torch.Size([1, 512])
    cosa = torch.matmul(face1_norm, face2_norm.T)
    # cosa = torch.dot(face1_norm.reshape(-1), face2_norm.reshape(-1))
    return cosa


if __name__ == '__main__':
    a = torch.randn(100, 3, 28, 28)
    net = FaceNet()
    net(a)
    # print(models.mobilenet_v2())
