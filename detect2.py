import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
import utils
import nets
from torchvision import transforms
import time
import os
import matplotlib.pyplot as plt


class Detector:
    def __init__(self, pnet_param="./param4/p_net.pth", rnet_param="./param4/r_net.pth", onet_param="./param4/o_net.pth",
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

        self.pnet.eval() # 批归一化, 使用之前训练的Batchnormal
        self.rnet.eval()
        self.onet.eval()

        self._image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )

    def detect(self, image):

        start_time = time.time()  #获取当前时间的函数。
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

        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

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
        idxs, _ = np.where(cls > 0.6)  # 0.7-0.8
        for idx in idxs:
            _box = _pnet_boxes[idx]  # 根据索引拿到裁剪框（预测框）的坐标，便于反算
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            # print(cls[idx][0])  # 拿到标量

            boxes.append([x1, y1, x2, y2, cls[idx][0]])

        return utils.nms(np.array(boxes), 0.5)

    def __onet_detect(self, image, rnet_boxes):

        _img_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
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
        idxs, _ = np.where(cls > 0.95)  # 一般为0.99
        for idx in idxs:
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1
            # 网络在输入裁剪图片的时候，坐标已经知道，无需反算预测框，直接反算预测框到真实框的位置即可。
            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]])

        return utils.nms(np.array(boxes), 0.7, isMin=True)  # #阈值0.7。保留IOU小于0.7的框

    def __pnet_detect(self, image):

        boxes = []
        img = image
        w, h = img.size  # 图片的宽高

        min_side_len = min(w, h)  # 得到图片最小的一边
        scale = 1  #缩放比例

        while min_side_len > 12:
            img_data = self._image_transform(img)
            if self.isCuda:
                img_data = img_data.cuda()

            # print(img_data.shape)  # torch.Size([3, 722, 1200])
            img_data.unsqueeze_(0)  # 加批次N, torch.Size([1, 3, 722, 1200])

            _cls, _offest = self.pnet(img_data)
            # print(_cls.shape)  # torch.Size([1, 1, 722, 1200])
            # print(_offest.shape)  # torch.Size([1, 4, 722, 1200])

            cls, offest = _cls[0][0].cpu().data, _offest[0].cpu().data  # _cls:H, W, _offset:C H W
            idxs = torch.nonzero(torch.gt(cls, 0.5), as_tuple=False)  # 置信度一般为0.5,先得到布尔值，再取出索引
            # print(idxs.shape)  # torch.Size([N, 2])

            for idx in idxs:  # 拿到所有满足要求的特征图索引，开始遍历，反算到原图的真实框
                boxes.append(self.__box(idx, offest, cls, scale))  # 添加符合条件的真实框, 拿到索引对应置信度的值。


            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            print(np.shape(img))
            min_side_len = np.minimum(_w, _h)

        return utils.nms(np.array(boxes), 0.3)  # 保留IOU小于0.5的框，IOU取值越小，去框越多。

    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        # 利用特征图的索引反算到原图上的预测框
        _x1 = (start_index[1] * stride).float() / scale
        _y1 = (start_index[0] * stride).float() / scale
        _x2 = (start_index[1] * stride + side_len).float() / scale
        _y2 = (start_index[0] * stride + side_len).float() / scale

        ow = _x2 - _x1
        oh = _y2 - _y1
        # 利用预测框位置反算到真实框。
        cls = cls[start_index[0], start_index[1]]  # 取到对应索引位置的置信度

        # print(offset.shape)  # torch.Size([4, 722, 1200])
        _offset = offset[:, start_index[0], start_index[1]]  # C H W 取H W对应位置
        # print(_offset)  # tensor([ 0.0773, -0.1806,  0.3268,  0.5450])

        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]


if __name__ == '__main__':
    x = time.time()
    with torch.no_grad() as grad:

        image_path = r"D:\PycharmProjects\MTCNN_data\picture2"
        detector = Detector()

        for file in os.listdir(image_path):
            img = Image.open("{0}/{1}".format(image_path, file))
            # print(img)
            # print(np.shape(img))
            # exit()

            boxes = detector.detect(img)
            # print(img.size)
            # print(img)

            imDraw = ImageDraw.Draw(img)

            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                print(box[4])
                imDraw.rectangle((x1, y1, x2, y2), outline='red', width=3)

            y = time.time()
            print(y - x)

                # im.show()
            plt.title("侦测人脸")
            plt.imshow(img)
            plt.pause(5)
