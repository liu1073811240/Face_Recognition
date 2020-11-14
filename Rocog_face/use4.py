from PIL import ImageDraw, ImageFont, Image
from Rocog_face.face import *
from Rocog_face.Mydataset import tf
import numpy as np
import cv2
import os
from Rocog_face.utils import trans_square, npz2list
import time


class using:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = r"D:\PycharmProjects\MTCNN_data\Rocog_face\params\1.pth"
        self.net = FaceNet().to(self.device)
        self.net.load_state_dict(torch.load(self.save_path))
        self.net.eval()

    def us(self, face_crop):  # face_crop
        # person1_feature = net.encode(person1[None, ...])
        # print(person1.shape)  # torch.Size([3, 112, 112])
        # print(torch.unsqueeze(person1, 0).shape)  # torch.Size([1, 3, 112, 112])
        # print(person1[None, ...].shape)  # torch.Size([1, 3, 112, 112])
        start = time.time()
        # print(np.shape(face_crop))  # (342, 258, 3)
        face_crop = trans_square(face_crop)
        # print(np.shape(face_crop))  # (342, 342, 3)
        # 将裁剪出来的图片转成正方形112*112
        person2 = tf(face_crop).to(self.device)
        # print(np.shape(face_crop))  # (342, 342, 3)

        # person2 = tf(Image.open("Contrast_data/1/pic1_0.jpg")).to(self.device)  # 需要辨认的人脸图片
        person2_feature = self.net.encode(person2[None, ...])

        # data_path = r"D:\PycharmProjects(2)\arcloss-pytorch\test_img"  # # 人脸数据库
        main_dir = r"D:\PycharmProjects\MTCNN_data\Rocog_face\Contrast_data"
        dicts = {"0": "周杰伦", "1": "迪丽热巴", "2": "黄晓明", "3": "刘辉", "4": "吴晓斌"}

        lists = []
        lists2 = []
        lists3 = []
        # for face_dir in os.listdir(main_dir):
        #     for face_filename in os.listdir(os.path.join(main_dir, face_dir)):
        #         img = Image.open(os.path.join(main_dir, face_dir, face_filename))
        #         img = trans_square(img)
        #
        #         # 将拿到的图片转成正方形112*112
        #         person1 = tf(img).to(self.device)
        #         person1_feature = self.net.encode(torch.unsqueeze(person1, 0))
        #         lists3.append([person1_feature, face_dir])

        # 改进： 改为并行比较，或查找方法，提高对比速度
        path = r"D:\PycharmProjects\MTCNN_data\Rocog_face\list_data.npz"
        list_o = npz2list(path)  # 载入numpy保存的人脸数据库文件
        # print(np.shape(list_o))
        for x in list_o:  # 遍历数据库中的人脸特征
            # print(x)
            # print(x[1])  # 0
            # print(x[0].shape)  # torch.Size([1, 512])
            siam = compare(x[0], person2_feature)  # 将数据库中的人脸和需要辨认的人脸做比较
            print("余弦相似度值：", max(siam.item(), 0))  # 余弦相似度 tensor([[0.9988]])
            # x = "周杰伦" if round(siam.item()) == 1 else "其他人"
            # x = "迪丽热巴" if round(siam.item()) == 1 else "其他人"
            # print(face_filename)

            # font = ImageFont.truetype("simhei.ttf", 20)

            if siam.item() > 0.8:  # 人脸相似度大于某个阈值，否则被pass掉。
                # print(x[1])  # 类别文件夹
                value = dicts[str(x[1])]
                lists.append(value)
                # print(lists)  # ['黄晓明', '黄晓明', '黄晓明', '黄晓明', '刘辉', '刘辉']
                # print(value)  # 将字典里的值取出来
                # print("*" * 100)
            else:
                pass

        end = time.time()
        print("对比完所有图片所用时间：", end - start)  # 0.75s
        # np.savez('list_data', lists3)
        # print(lists1)  # 每比较完一类图片，打印一次列表
        # print(lists3)
        return lists


if __name__ == '__main__':
    img = Image.open(r"D:\PycharmProjects\MTCNN_data\Rocog_face\face_images\1.jpg")
    u = using()
    u.us(img)
    # 把模型和参数进行打包，以便C++或PYTHON调用
    # import torch.jit as jit
    # x = torch.Tensor(1, 3, 112, 112)
    # net = FaceNet()
    # net.load_state_dict(torch.load("params/1.pt"))
    # net.eval()
    # traced_script_module = jit.trace(net, x)
    # traced_script_module.save("model.cpt")
