from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import shutil
from Rocog_face.utils import trans_square
import numpy as np

tf = transforms.Compose([
    transforms.Resize([112, 112]),  # 不会成比例缩放，图片特征会变形，所以输入图片一定要先转成正方形（缺少的区域填充黑色）
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class MyDataset(Dataset):
    def __init__(self, main_dir):
        self.dataset = []
        for face_dir in os.listdir(main_dir):
            for face_filename in os.listdir(os.path.join(main_dir, face_dir)):
                # print(main_dir, face_dir, face_filename)  # Contrast_data 0 pic100_0.jpg
                self.dataset.append([os.path.join(main_dir, face_dir, face_filename), int(face_dir)])
                # print(self.dataset)  # [['人脸图片路径', 0],...]

    def __len__(self):
        return len(self.dataset)  # 返回有多少张人脸

    def __getitem__(self, item):
        data = self.dataset[item]
        # image_data = tf(Image.open(data[0]))  # 打开对应图片并转换
        image = Image.open(data[0])
        # print(np.shape(image))  # (268, 267, 3)
        image_data = trans_square(image)
        # print(np.shape(image_data))  # (268, 268, 3)
        image_data = tf(image_data)
        # print(np.shape(image_data))  # torch.Size([3, 112, 112])

        image_label = data[1]  # 拿到对应图片标签
        return image_data, image_label


if __name__ == '__main__':

    mydataset = MyDataset("face_data")
    dataset = DataLoader(mydataset, 100, shuffle=True)
    for data in dataset:
        print(data[0].shape)  # torch.Size([100, 3, 112, 112])
        print(data[1].shape)  # torch.Size([100])
        print(len(data[1]))  # 100
