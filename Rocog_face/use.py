from PIL import ImageDraw,ImageFont,Image
from Rocog_face.face2 import *
from Rocog_face.Mydataset import tf
import numpy as np
import cv2

class using:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = r"D:\PycharmProjects\MTCNN_data\Rocog_face\params\3.pth"
        self.net = FaceNet().to(self.device)
        self.net.load_state_dict(torch.load(self.save_path))
        self.net.eval()

    def us(self, face_crop):  # face_crop

        data_path = r"D:\PycharmProjects\MTCNN_data\Rocog_face\face_images\database_face_1.jpg"  # # 人脸数据库
        img = Image.open(data_path)
        img = img.convert("RGB")

        person1 = tf(img).to(self.device)
        person1_feature = self.net.encode(torch.unsqueeze(person1, 0))
        # person1_feature = net.encode(person1[None, ...])
        # print(person1.shape)  # torch.Size([3, 112, 112])
        # print(torch.unsqueeze(person1, 0).shape)  # torch.Size([1, 3, 112, 112])
        # print(person1[None, ...].shape)  # torch.Size([1, 3, 112, 112])

        person2 = tf(face_crop).to(self.device)
        # person2 = tf(Image.open("./face_images/recog_face_1.jpg")).to(self.device)  # 需要辨认的人脸图片
        person2_feature = self.net.encode(person2[None, ...])

        siam = compare(person1_feature, person2_feature)
        print("余弦相似度值：", siam.item())  # 余弦相似度 tensor([[0.9988]])
        # x = "周杰伦" if round(siam.item()) == 1 else "其他人"
        # x = "迪丽热巴" if round(siam.item()) == 1 else "其他人"
        x = "Liu Hui" if siam.item() >= 0.8 else "other people"

        return x
        # font = ImageFont.truetype("simhei.ttf", 20)
        # with Image.open("face_images/recog_face_1.jpg") as img:
        #     imgdraw = ImageDraw.Draw(img)
        #     imgdrawa = imgdraw.text((0, 0), x, font=font)
        #     img.show(imgdrawa)
        # print()


if __name__ == '__main__':
    u = using()
    # u.us()
    # 把模型和参数进行打包，以便C++或PYTHON调用
    # import torch.jit as jit
    # x = torch.Tensor(1, 3, 112, 112)
    # net = FaceNet()
    # net.load_state_dict(torch.load("params/1.pt"))
    # net.eval()
    # traced_script_module = jit.trace(net, x)
    # traced_script_module.save("model.cpt")
