import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
import utils
import nets
from torchvision import transforms
import time
import cv2
from Rocog_face.use3 import using
from Rocog_face.convert_words import change_cv2_draw
from Rocog_face.utils import cv2ImgAddText

class Detector:
    def __init__(self, pnet_param="./param6/p_net.pth", rnet_param="./param6/r_net.pth",
                 onet_param="./param6/o_net.pth",
                 isCuda=True):

        self.isCuda = isCuda
        self.pnet = nets.PNet()
        self.rnet = nets.RNet()
        self.onet = nets.ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()
        self.pnet.load_state_dict(torch.load(pnet_param, map_location="cuda"))
        self.rnet.load_state_dict(torch.load(rnet_param, map_location="cuda"))
        self.onet.load_state_dict(torch.load(onet_param, map_location="cuda"))

        self.pnet.eval()  # 批归一化, 使用之前训练的Batchnormal
        self.rnet.eval()
        self.onet.eval()

        self._image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )

    def detect(self, image):

        start_time = time.time()  # 获取当前时间的函数。
        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:  # 防止程序格式错误
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time
        # return pnet_boxes

        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        # print( rnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time

        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        t_sum = t_pnet + t_rnet + t_onet

        # print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes  # 可以更改为p网络框进行测试

    def __rnet_detect(self, image, pnet_boxes):

        _img_dataset = []
        _pnet_boxes = utils.convert_to_square(pnet_boxes)

        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self._image_transform(img)

            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.rnet(img_dataset)
        cls = _cls.cpu().data.numpy()

        offset = _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(cls > 0.7)  # 0.6、0.7、0.8

        _box = _pnet_boxes[idxs]
        # print(_box.shape)  # (16, 5)
        _x1 = np.array(_box[:, 0])
        _y1 = np.array(_box[:, 1])
        _x2 = np.array(_box[:, 2])
        _y2 = np.array(_box[:, 3])

        ow = _x2 - _x1
        oh = _y2 - _y1

        offset = offset[idxs].T
        cls = cls[idxs].T

        x1 = _x1 + ow * offset[0, :]  # 偏移框反算到真实框
        y1 = _y1 + oh * offset[1, :]
        x2 = _x2 + ow * offset[2, :]
        y2 = _y2 + oh * offset[3, :]

        out_boxes = np.dstack((x1, y1, x2, y2, cls))
        # print(out_boxes.shape)  # (1, 16, 5)
        out_boxes = np.squeeze(out_boxes, 0)
        # print(out_boxes.shape)  # (16, 5)

        return utils.nms(np.array(out_boxes), 0.5)  # 0.5

    def __onet_detect(self, image, rnet_boxes):

        _img_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        # print(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self._image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.onet(img_dataset)

        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(cls > 0.99)  # 一般为0.99

        _box = rnet_boxes[idxs]
        # print(_box.shape)  # (16, 5)
        _x1 = np.array(_box[:, 0])
        _y1 = np.array(_box[:, 1])
        _x2 = np.array(_box[:, 2])
        _y2 = np.array(_box[:, 3])

        ow = _x2 - _x1
        oh = _y2 - _y1

        offset = offset[idxs].T
        cls = cls[idxs].T

        x1 = _x1 + ow * offset[0, :]  # 偏移框反算到真实框
        y1 = _y1 + oh * offset[1, :]
        x2 = _x2 + ow * offset[2, :]
        y2 = _y2 + oh * offset[3, :]

        out_boxes = np.dstack((x1, y1, x2, y2, cls))
        # print(out_boxes.shape)  # (1, 16, 5)
        out_boxes = np.squeeze(out_boxes, 0)
        # print(out_boxes.shape)  # (16, 5)

        return utils.nms(np.array(out_boxes), 0.7, isMin=True)  # #阈值0.7。保留IOU小于0.7的框

    def __pnet_detect(self, image):
        # start_time = time.time()
        boxes = np.array([[0, 0, 0, 0, 0]])
        # boxes = []
        img = image
        w, h = img.size  # 图片的宽高

        min_side_len = min(w, h)  # 得到图片最小的一边
        scale = 0.6  # 缩放比例一般为1

        while min_side_len > 12:
            img_data = self._image_transform(img)
            if self.isCuda:
                img_data = img_data.cuda()

            # print(img_data.shape)  # torch.Size([3, 722, 1200])
            img_data.unsqueeze_(0)  # 加批次N, torch.Size([1, 3, 722, 1200])

            _cls, _offset = self.pnet(img_data)
            # print(_cls.shape)  # torch.Size([1, 1, 722, 1200])
            # print(_offest.shape)  # torch.Size([1, 4, 722, 1200])

            # cls, offset = _cls[0][0].cpu().data, _offest[0].cpu().data  # _cls:H, W, _offset:C H W
            cls, offset = _cls[0][0].cpu().data, _offset[0].cpu().data
            # print(cls)
            # print(_offset.shape)  # torch.Size([1, 4, 814, 1076])
            # exit()

            idxs = torch.nonzero(torch.gt(cls, 0.6), as_tuple=False)  # 置信度一般为0.5,先得到布尔值，再取出索引

            # for idx in idxs:  # 拿到所有满足要求的特征图索引，开始遍历，反算到原图的真实框
            # boxes.append(self.__box(idx, offset, cls, scale))  # 添加符合条件的真实框, 拿到索引对应置信度的值。

            boxes = np.concatenate((boxes, self.__box(idxs, offset, cls, scale)), axis=0)  # 在横轴上进行拼接

            scale *= 0.5  # 图像金字塔缩放比例0.3~0.7
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_side_len = np.minimum(_w, _h)

        # end_time = time.time()
        # print("侦测P网络所用时间：", end_time-start_time)

        return utils.nms(np.array(boxes), 0.3)  # 保留IOU小于0.3的框，IOU取值越小，去框越多。

    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        # 利用特征图的索引反算到原图上的预测框
        _x1 = (start_index[:, 1] * stride) / scale
        _y1 = (start_index[:, 0] * stride) / scale
        _x2 = (start_index[:, 1] * stride + side_len) / scale
        _y2 = (start_index[:, 0] * stride + side_len) / scale
        # print(_x1.shape)  # torch.Size([103])

        ow = _x2 - _x1
        # print(ow.shape)  # torch.Size([103])
        oh = _y2 - _y1
        # 利用预测框位置反算到真实框。
        cls = cls[start_index[:, 0], start_index[:, 1]]  # 取到对应索引位置的置信度
        # print(cls.shape)  # torch.Size([103])

        _offset = offset[:, start_index[:, 0], start_index[:, 1]]  # C H W 取H W对应位置
        # print(_offset[0, :].shape)  # torch.Size([4, 103])

        x1 = _x1 + ow * _offset[0, :]
        y1 = _y1 + oh * _offset[1, :]
        x2 = _x2 + ow * _offset[2, :]
        y2 = _y2 + oh * _offset[3, :]

        out_boxes = np.dstack((x1, y1, x2, y2, cls))  # 纵向堆叠数据
        # print(out_boxes)
        # print(out_boxes.shape)  # (1, 103, 5)
        out_boxes = np.squeeze(out_boxes, 0)
        # print(out_boxes)
        # print(out_boxes.shape)  # (103, 5)

        return out_boxes

from Rocog_face.face import *
from Rocog_face.Mydataset import tf
import os

class FaceDetector:
    def __init__(self):
        path = r"Rocog_face/Contrast_data"
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
        # print('----------完美分割线----------------')
        if max_threshold > 0:
            name = self.face_dict[max_threshold_feature]
            y = time.time()
            # print(y - x)
            return name,max_threshold.item()
        return '','0.0'


# from Rocog_face.use5 import FaceDetector
if __name__ == '__main__':

    with torch.no_grad() as grad:
        # face_detector = using()
        face_detector = FaceDetector()

        # path = "1.mp4"  # 本地视频路径
        # path = r"http://vfx.mtime.cn/Video/2019/03/19/mp4/190319125415785691.mp4"  # 在线视频路径
        # cap = cv2.VideoCapture(path)
        cap = cv2.VideoCapture(0)  # 调取内置摄像头
        # cap.set(3, 120)
        # cap.set(4, 160)

        # fps = cap.get(cv2.CAP_PROP_FPS)
        # print(fps)

        w = int(cap.get(3))  # 获取图片的宽度
        h = int(cap.get(4))  # 获取图片的高度
        # print(w)  # 640
        # print(h)  # 480

        # 写入视频格式
        # fourc = cv2.VideoWriter_fourcc(*"DVIX")
        # out = cv2.VideoWriter("2.mp4", fourc, fps, (w, h))

        font = cv2.FONT_HERSHEY_COMPLEX
        # frame表示读出的每一张图片, ret表示这一张图片是否存在
        # fourc = cv2.VideoWriter_fourcc(*"MJPG")  # 视频格式
        # out = cv2.VideoWriter("2.mp4", fourc, fps, (w, h))  # 写入视频格式
        count = 0
        lists = []
        lists1 = []
        lists2 = []
        text2 = []
        result = []
        while True:
            a = time.time()
            ret, frame1 = cap.read()  # 该步所用时间0.0079

            # 将十六进制数据转成 二进制数据
            if cv2.waitKey(int(1)) & 0xFF == ord("q"):  # 视频在播放的过程中按键，循环会中断）。
                break
            elif ret == False:  # 视频播放完了，循环自动中断。
                break

            # frame2 = Image.fromarray(frame1)
            frame2 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
            width, height = frame2.size  # 640  480
            # 将传给网络图片的变小，侦测速度会变快
            frame3 = frame2.resize((int(width * 0.1), int(height * 0.1)), Image.ANTIALIAS)

            # print("时间：", y-x)

            detector = Detector()
            boxes = detector.detect(frame3)

            # print("侦测出来的框：", boxes)

            count_b = 0
            for box in boxes:  # 遍历这一帧中有多少个人脸框

                # w = int(box[2] - box[0])
                # h = int(box[3] - box[1])
                # x1 = int((box[0]) - 0.2 * w)
                # y1 = int((box[1]))
                # x2 = int((box[2]) - 0.1 * w)
                # y2 = int((box[3]) - 0.2 * h)

                # x1 = int(box[0] - 0.2*w)
                # y1 = int(box[1])
                # x2 = int(box[2] - 0.1*w)
                # y2 = int(box[3] - 0.2*h)

                w = int((box[2] - box[0]) / 0.1)  # 将缩小的图片以比例反算回原图大小
                h = int((box[3] - box[1]) / 0.1)
                x1 = int((box[0]) / 0.1 - 0.2*w)
                y1 = int((box[1]) / 0.1)
                x2 = int((box[2]) / 0.1 - 0.1*w)
                y2 = int((box[3]) / 0.1 - 0.2*h)

                # print("侦测该张图片的目标置信度", box[4])
                cv2.rectangle(frame1, (x1, y1), (x2, y2), [0, 0, 255], 1)

                # 注意： 如果检测出来的图片比较暗，就需要用opencv进行处理，然后再传到识别网络进行识别。
                face_crop = frame2.crop((x1, y1, x2, y2))  # 将侦测到的人脸裁剪下来以便传到分类网络
                # print(np.shape(face_crop))  # (325, 229, 3)
                # face_crop.save(r"D:\PycharmProjects\MTCNN_data\Rocog_face\save_face_crop\{}.jpg".format(count))

                # lists = using().us(face_crop)
                # if len(lists) == 0:
                #     pass
                # else:
                #     text = str(lists[-1])
                #     text2.append(text)
                #     print(text2)  # ['刘辉', '刘辉']
                #     # cv2.putText(frame1, text, (x1, y1), font, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                #     frame1 = cv2ImgAddText(frame1, text, x1, y1, (255, 0, 0), 20)  # 中文

                # dicts = {"0": "周杰伦", "1": "迪丽热巴", "2": "黄晓明", "3": "刘辉", "4": "目标未识别", "5": "吴京", "6": "张泽"}
                dicts = {"0": "其他人", "1": "其他人", "2": "其他人", "3": "刘辉", "4": "其他人", "5": "其他人", "6": "其他人"}
                # dicts = {"0": "安然", "1": "大力娇"}
                x = time.time()
                name, max_threshold = face_detector.face_detector(face_crop)  # 用时0.04s

                y = time.time()
                print(y-x)

                # 普通侦测
                if len(name) == 0:
                    pass
                else:
                    value = dicts[str(name)]
                    lists.append(value)

                    frame1 = cv2ImgAddText(frame1, value, x1, y1, (255, 0, 0), 40)

                # 侦测一张人脸
                # if count % 30 == 0:
                #     if len(name) == 0:
                #         pass
                #     else:
                #         value = dicts[str(name)]
                #         lists.append(value)
                #         frame1 = cv2ImgAddText(frame1, value, x1, y1, (255, 0, 0), 40)
                # else:
                #     if len(lists) == 0:
                #         pass
                #     else:
                #         value = str(lists[-1])
                #         frame1 = cv2ImgAddText(frame1, value, x1, y1, (255, 0, 0), 40)

                # 侦测两张人脸

                # if count_b % 2 == 0:  # 侦测两个目标
                #     if count % 30 == 0:
                #         if len(name) == 0:
                #             pass
                #         else:
                #             value = dicts[str(name)]
                #             lists1.append(value)
                #             print("lists1", lists1)
                #             frame1 = cv2ImgAddText(frame1, value, x1, y1, (255, 0, 0), 40)
                #     else:
                #         if len(lists1) == 0:
                #             pass
                #         else:
                #             value = str(lists1[-1])
                #             frame1 = cv2ImgAddText(frame1, value, x1, y1, (255, 0, 0), 40)
                # else:
                #     if count % 30 == 0:
                #         if len(name) == 0:
                #             pass
                #         else:
                #             value = dicts[str(name)]
                #             lists2.append(value)
                #             print("lists2", lists2)
                #             frame1 = cv2ImgAddText(frame1, value, x1, y1, (255, 0, 0), 40)
                #     else:
                #         if len(lists2) == 0:
                #             pass
                #         else:
                #             value = str(lists2[-1])
                #             frame1 = cv2ImgAddText(frame1, value, x1, y1, (255, 0, 0), 40)

                # print(count_b)



                # if count % 30 == 0:
                #     names = []
                #     name, max_threshold = face_detector.face_detector(face_crop)
                #     print('name', name)
                #     frame1 = cv2ImgAddText(frame1, name, x1, y1, (255, 0, 0), 40)
                #     names.append(name)
                # else:
                #     for name in names:
                #         frame1 = cv2ImgAddText(frame1, name, x1, y1, (255, 0, 0), 40)
                # if count % 30 == 0:  # 每八帧做一次人脸分类
                #     text2 = []
                #     lists = using().us(face_crop)
                #     name, max_threshold = face_detector.face_detector(face_crop)
                #     print('------------------------- ',lists)
                #     if len(lists) == 0:  # 防止列表为空报错
                #         pass
                #     else:
                #         text = str(lists[-1])
                #         text2.append(text)
                #         print(text2)  # ['刘辉', '刘辉']
                #         # cv2.putText(frame1, text, (x1, y1), font, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                #         frame1 = cv2ImgAddText(frame1, text, x1, y1, (255, 0, 0), 20)  # 中文
                #
                # else:  # 直接拿列表里的文本信息，就不用再经过分类网络浪费时间
                #     if len(text2) == 0:
                #         pass
                #     else:
                #         text3 = str(text2[-1])
                #         # print("text2", text2)  # ['刘辉', '刘辉', '目标未识别']
                #         # print("text3", text3)
                #         # cv2.putText(frame1, text3, (x1, y1), font, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                #         frame1 = cv2ImgAddText(frame1, text3, x1, y1, (255, 0, 0), 20)

            count = count + 1  # 每过完一帧计数一次
            # print(count)
            b = time.time()
            # print("侦测一张图片所用时间：", y - x)
            print("FPS:", 1/(b-a))

            # out.write(frame1)

            cv2.imshow("Detect", frame1)

        cap.release()  # 将视频关了
        cv2.destroyAllWindows()