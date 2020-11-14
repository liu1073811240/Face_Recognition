from Rocog_face.face import *
from Rocog_face.Mydataset import tf
import os
import time
from PIL import Image
from Rocog_face.utils import trans_square, npz2list
import numpy as np


class FaceDetector:
    def __init__(self):
        path = r"D:\PycharmProjects\MTCNN_data\Rocog_face\Contrast_data"
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
        self.lists3 = []

        # for face_dir in os.listdir(path):
        #     for face_filename in os.listdir(os.path.join(path, face_dir)):
        #         img = Image.open(os.path.join(path, face_dir, face_filename))
        #
        #         img = trans_square(img)
        #         person1 = tf(img).to(device)
        #         person1_feature = self.net.encode(torch.unsqueeze(person1, 0))
        #         self.face_dict[person1_feature] = face_dir  # 将人脸特征向量作为键，类别名作为值
        #         # print(self.face_dict[person1_feature])  # 人脸特征向量对应类别名 0
        #         self.lists3.append([person1_feature, face_dir])  # 将人脸特征向量保存下来

        # np.savez('list_data', self.lists3)
    def face_detector(self, img):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        max_threshold = 0
        threshold = 0.7
        max_threshold_feature = 0
        name = 0
        person1 = tf(img).to(device)  # 需要辨认的人脸
        person1_feature = self.net.encode(torch.unsqueeze(person1, 0))

        # 载入np人脸特征向量包
        np_path = r"D:\PycharmProjects\MTCNN_data\Rocog_face\list_data.npz"
        list_o = npz2list(np_path)  # 载入numpy保存的人脸数据库文件
        # print(np.shape(list_o))
        for x in list_o:  # 遍历数据库中的人脸特征
            # print(x)
            siam = compare(x[0], person1_feature)  # 将数据库中的人脸和需要辨认的人脸做比较
            # print("余弦相似度值：", max(siam.item(), 0))  # 余弦相似度 tensor([[0.9988]])

            if siam > threshold and siam > max_threshold:
                max_threshold = siam  # 如果余弦相似度大于所设置阈值，就赋值给最大阈值。同时能够更新所设置的最大阈值。因为最后只需要拿到余弦相似度最大的哪个值。
                max_threshold_feature = x[0]
                name = x[1]

        # print('----------完美分割线----------------')
        # if max_threshold > 0:
        #     print(max_threshold_feature)
        #     name = self.face_dict[max_threshold_feature]  # 拿到余弦相似度最大时的人脸特征向量对应的类别名
            # print(max_threshold_feature)  # 拿到余弦相似度最大时的人脸特征向量

        # print(name)
        # print(max_threshold_feature)

        return name, max_threshold_feature

        # return '', '0.0'


if __name__ == '__main__':
    x = time.time()
    img = Image.open(r"D:\PycharmProjects\MTCNN_data\Rocog_face\face_images\1.jpg")
    u = FaceDetector()
    u.face_detector(img)

    y = time.time()
    print(y - x)