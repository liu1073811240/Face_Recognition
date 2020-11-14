import torchvision.models as models
from torch import nn
import torch
from torch.nn import functional as F
# from deep.homework.face_disting.face_dataset import *
# from deep.homework.face_disting.net.FaceNet import FaceNet
# from deep.homework.face_disting.tool import utils
from torch import optim
from torch.utils.data import DataLoader
import torch.jit as jit
import time
from PIL import ImageDraw, ImageFont, Image
from Rocog_face.face import *
from Rocog_face.Mydataset import tf
import numpy as np
import cv2
import os
from Rocog_face.utils import trans_square
import time

class FaceDetector:
    def __init__(self):
        path = r"Contrast_data"
        # self.save_path = r"D:\PycharmProjects\MTCNN_data\Rocog_face\params\1.pth"
        net_path = r"D:\PycharmProjects\MTCNN_data\Rocog_face\params\1.pth"
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.net = FaceNet().to(device)
        self.net.load_state_dict(torch.load(net_path))
        self.net.eval()
        self.face_dict = {}

        for face_dir in os.listdir(path):
            for face_filename in os.listdir(os.path.join(path, face_dir)):
                person_path = os.path.join(path, face_dir, face_filename)
                img = Image.open(person_path)
                img = img.convert("RGB")
                img = img.resize((112, 112))
                person1 = tf(img).to(device)
                person1_feature = self.net.encode(torch.unsqueeze(person1, 0))
                self.face_dict[person1_feature] = face_dir
    def face_detector(self,img):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        max_threshold = 0
        threshold = 0.7
        max_threshold_feature = 0
        person1 = tf(img).to(device)
        person1_feature = self.net.encode(torch.unsqueeze(person1, 0))
        kys = self.face_dict.keys()
        kys = list(kys)

        # print(kys[0].shape)
        # a = torch.randn([len(kys),kys[0].shape[0],kys[0].shape[1]])
        # print(a.shape)
        # print(a[0].shape)
        # exit()

        for person_feature in kys:
            # print(person_feature.shape)
            siam = compare(person1_feature, person_feature)
            # print(self.face_dict[person_feature], siam)
            if siam > threshold and siam > max_threshold:
                max_threshold = siam
                max_threshold_feature = person_feature
        print('----------完美分割线----------------')
        if max_threshold > 0:
            name = self.face_dict[max_threshold_feature]
            y = time.time()
            # print(y - x)
            return name,max_threshold.item()
        return '','0.0'

if __name__ == '__main__':

    with torch.no_grad() as grad:
        face_detector = FaceDetector()
        path = r"face_images"
        for face_filename in os.listdir(path):
            person_path = os.path.join(path, face_filename)
            img = Image.open(person_path)
            img = img.convert("RGB")
            img = img.resize((112, 112))
            print('========== ',person_path)
            name,max_threshold = face_detector.face_detector(img)
            print(person_path,name,max_threshold)
        x = time.time()
        # person_path = r"test_face_pic/5.jpg"
        # img = Image.open(person_path)
        # name = face_detector.face_detector(img)
        # print(person_path, name)
        # y = time.time()
        # print(y - x)

