import cv2
import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
from tool import utils
import nets
from torchvision import transforms
import time
import detdct_tow

if __name__ == '__main__':
    # path = r"C:\Users\lieweiai\Desktop\5、opencv(1)\test20200727\1.mp4"
    # path = r"C:\Users\lieweiai\Desktop\mtcnn\VID_20200923_094038.mp4"
    # cap = cv2.VideoCapture(path)
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(1400)
    while True:
        start_time = time.time()
        ret, frame = cap.read()  # 将每一帧都读出来
        if cv2.waitKey(1) & 0xFF == ord("q"):  # int(1000/fps)决定播放速度，必须为整数，为零则是按住空格播放，为1则是实时播放  0xFF==ord("q")：给个按键q退出
            break
        elif ret == False:
            break  # 如果没有elif，则程序在执行玩前面后会报错，终止后面的程序运行

        with torch.no_grad() as grad:
            image_file = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            # image_file = cv2.imread(frame)
            detector = detdct_tow.Detector(isCuda=True)
            # x = time.time()
            # print(im)
            boxes = detector.detect(image_file)
            # print(boxes.shape)
            # imDraw = ImageDraw.Draw(image_file)
            for box in boxes:

                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                # print(x1)
                # imDraw.rectangle((x1, y1, x2, y2), outline='red', width=3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 3)  # 在视频图像里画矩阵框
                # y = time.time()

            # im.show()

        cv2.imshow("", frame)
        end_time = time.time()
        print(1/(end_time - start_time))
        # print(image_file.size)
    cap.release()
    cv2.destroyAllWindows()

