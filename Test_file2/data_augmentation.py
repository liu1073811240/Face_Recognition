import cv2
import os
import random
import numpy as np


def addImage(img1_path, img2_path, count):
    img1 = cv2.imread(img1_path)
    img = cv2.imread(img2_path)
    h, w, _ = img1.shape
    # 函数要求两张图必须是同一个size
    img2 = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA)
    #print img1.shape, img2.shape
    #alpha，beta，gamma可调
    alpha = 0.7
    beta = 1-alpha
    gamma = 0
    img_add = cv2.addWeighted(img1, alpha, img2, beta, gamma)

    # cv2.imshow('img_add',img_add)
    cv2.imwrite(r"D:\ACelebA\Gen_image2\{0}.jpg".format(str(count).zfill(6)), img_add)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def write_txt(count):
    with open(r"D:\ACelebA\Gen.txt", "r") as f1:
        Read_data = f1.readlines()
    f2 = open(r"D:\ACelebA\Gen2.txt", "a")
    for x in Read_data:
        # print(x)
        strs = x.split()
        # print(strs[0])
        if image == strs[0]:
            f2.write("{0}.jpg {1} {2} {3} {4}".format(str(count).zfill(6), strs[1], strs[2], strs[3], strs[4]))
            f2.write('\n')

    f2.close()


count = 10000

image_dir = r"D:\ACelebA\Gen_image"
Bg_image_dir = r"D:\ACelebA\Bg_image2"

for image in os.listdir(image_dir):
    print(image)  # 100.jpg
    for Bg_image in os.listdir(Bg_image_dir):
        img_path = os.path.join(image_dir, image)
        Bg_img_path = os.path.join(Bg_image_dir, Bg_image)

        # 合成图片
        # print(count)

        addImage(img_path, Bg_img_path, count)
        write_txt(count)

        # 加高斯模糊图片
        img_add = cv2.imread(r"D:\ACelebA\Gen_image\{0}".format(image))  # str(count).zfill(6)

        img_blur = cv2.GaussianBlur(img_add, (7, 7), 8)
        count += 1
        # print(count)
        cv2.imwrite(r"D:\ACelebA\Gen_image2/{0}.jpg".format(str(count).zfill(6)), img_blur)
        write_txt(count)  # 写入相应图片标签


        # with open(r"D:\CelebA\list_bbox_celeba2.txt", "r") as f1:
        #     Read_data = f1.readlines()
        # f2 = open(r"D:\PycharmProjects\MTCNN3_key_point\img.txt", "w")
        # count_txt = 0
        # for x in Read_data:
        #     print(x)
        #     strs = x.split()
        #     print(strs[0])
        #     if image == strs[0]:
        #         f2.write(x)
        # exit()

        # 加噪点、模糊图片

        pic = cv2.imread(r"D:\ACelebA\Gen_image\{0}".format(image))

        for i in range(1000):
            pic[random.randint(0, pic.shape[0] - 1)][random.randint(0, pic.shape[1] - 1)][:] = 0
        count += 1
        # print(count)
        cv2.imwrite(r"D:\ACelebA\Gen_image2/{0}.jpg".format(str(count).zfill(6)), pic)
        write_txt(count)

        # 加噪点、模糊、变暗图片
        pic2 = cv2.imread(r"D:\ACelebA\Gen_image\{0}".format(image))
        count += 1
        contrast = 1  # 对比度
        brightness = -100  # 亮度
        pic_turn = cv2.addWeighted(pic2, contrast, pic2, 0, brightness)
        # print(count)
        cv2.imwrite(r"D:\ACelebA\Gen_image2/{0}.jpg".format(str(count).zfill(6)), pic_turn)
        write_txt(count)

        count += 1




