from PIL import ImageDraw, ImageFont, Image
from Rocog_face.face import *
from Rocog_face.Mydataset import tf
import numpy as np
import cv2
import os
from Rocog_face.utils import trans_square
import time


class using:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = r"D:\PycharmProjects\MTCNN_data\Rocog_face\params\1.pth"
        self.net = FaceNet().to(self.device)
        self.net.load_state_dict(torch.load(self.save_path))
        self.net.eval()
        self.face_dict = {}

        x = time.time()
        main_dir = r"D:\PycharmProjects\MTCNN_data\Rocog_face\Contrast_data2"
        for face_dir in os.listdir(main_dir):
            for face_filename in os.listdir(os.path.join(main_dir, face_dir)):
                img = Image.open(os.path.join(main_dir, face_dir, face_filename))
                img = trans_square(img)

                # 将拿到的图片转成正方形112*112
                person1 = tf(img).to(self.device)
                person1_feature = self.net.encode(torch.unsqueeze(person1, 0))
                self.face_dict[person1_feature] = face_dir  # 将人脸特征向量作为键，类别名作为值
                # print(self.face_dict[person1_feature])  # 人脸特征向量对应类别名 0
                # exit()

                # lists3.extend(person1_feature)
                # lists3.append([person1_feature, face_dir])  # 方便将人脸数据库转成特征向量进行打包。

                # 改进： 改为并行比较，或查找方法，提高对比速度
                # print(person1_feature.shape)  # torch.Size([1, 512])
                # print(person2_feature.shape)  # torch.Size([1, 512])
        y = time.time()
        print(y-x)

    def us(self, face_crop):  # face_crop



        # data_path = r"D:\PycharmProjects(2)\arcloss-pytorch\test_img"  # # 人脸数据库

        # dicts = {"0": "周杰伦", "1": "迪丽热巴", "2": "黄晓明", "3": "刘辉", "4": "目标未识别", "5": "小红", "6": "小花"}

        # print(np.shape(face_crop))  # (342, 258, 3)

        face_crop = trans_square(face_crop)

        # print(np.shape(face_crop))  # (342, 342, 3)
        # 将裁剪出来的图片转成正方形112*112
        person2 = tf(face_crop).to(self.device)
        # print(np.shape(face_crop))  # (342, 342, 3)

        # person2 = tf(Image.open("Contrast_data/1/pic1_0.jpg")).to(self.device)
        person2_feature = self.net.encode(person2[None, ...])  # 传进来的人脸图片

        kys = self.face_dict.keys()  # 拿到人脸数据库里面的键（即人脸特征向量）
        # print(self.face_dict)
        # print(kys)
        kys = list(kys)  # 将人脸特征向量放在一个列表中
        # print(kys)

        # siam = compare(person1_feature, person2_feature)  # 将数据库中的人脸和需要辨认的人脸做比较

        # print("余弦相似度值：", max(siam.item(), 0))  # 余弦相似度 tensor([[0.9988]])
                # x = "周杰伦" if round(siam.item()) == 1 else "其他人"
                # x = "迪丽热巴" if round(siam.item()) == 1 else "其他人"
                # print(face_filename)

                # font = ImageFont.truetype("simhei.ttf", 20)

        max_threshold = 0
        threshold = 0.7
        max_threshold_feature = 0

        for person1_feature_ in kys:  # 遍历数据库里面的人脸特征向量
            # print(person1_feature_)

            siam = compare(person1_feature_, person2_feature)
            if siam > threshold and siam > max_threshold:
                max_threshold = siam  # 如果余弦相似度大于所设置阈值，就赋值给最大阈值。同时能够更新所设置的最大阈值。因为最后只需要拿到余弦相似度最大的哪个值。
                max_threshold_feature = person1_feature_   # 这时也拿到相应的人脸特征向量

        if max_threshold > 0:
            cls = self.face_dict[max_threshold_feature]  # 拿到余弦相似度最大时的人脸特征向量对应的类别名
            # print(max_threshold_feature)  # 拿到余弦相似度最大时的人脸特征向量

            return cls, max_threshold_feature

        return '', '0.0'  # 如果上面返回的是空值， 此时要设置默认值占位。


if __name__ == '__main__':
    x = time.time()
    img = Image.open(r"D:\PycharmProjects\MTCNN_data\Rocog_face\face_images\1.jpg")
    u = using()
    u.us(img)

    y = time.time()
    print(y - x)
    # 把模型和参数进行打包，以便C++或PYTHON调用
    # import torch.jit as jit
    # x = torch.Tensor(1, 3, 112, 112)
    # net = FaceNet()
    # net.load_state_dict(torch.load("params/1.pt"))
    # net.eval()
    # traced_script_module = jit.trace(net, x)
    # traced_script_module.save("model.cpt")
