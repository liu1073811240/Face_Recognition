import os
from PIL import Image
import numpy as np
import utils
import traceback

anno_src = r"D:\ACelebA\Gen2.txt"
img_dir = r"D:\ACelebA\Gen_image2"

save_path = r"D:\ACelebA\Gen_image3"
for face_size in [48]:

    print("gen %i image" % face_size)
    # 样本图片存储路径
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")  # D:\CelebA\48\positive
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")  # D:\CelebA\48\negative
    part_image_dir = os.path.join(save_path, str(face_size), "part")  # D:\CelebA\48\part

    # 如果文件夹不存在，就创建相应文件夹。
    for image_dir in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

    # 样本标签存储路径
    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")  # D:\CelebA\48\positive.txt
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")  # D:\CelebA\48\negative.txt
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")  # D:\CelebA\48\part.txt

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:
        positive_anno_file = open(positive_anno_filename, "w")  # 打开正样本txt文件，开始准备写入标签文件
        negative_anno_file = open(negative_anno_filename, "w")  # # 打开负样本txt文件，开始准备写入标签文件
        part_anno_file = open(part_anno_filename, "w")  # 打开部分样本txt文件，开始准备写入标签文件

        for i, line in enumerate(open(anno_src)):  # 打开原始标签txt文件开始进行枚举, i表示索引第几行，line表示相应行的数据。

            if i < 2:  # 跳过前面两行数据
                continue
            try:
                strs = line.strip().split(" ")  # ['000001.jpg', '', '', '', '95', '', '71', '226', '313']
                strs = list(filter(bool, strs))  # ['000001.jpg', '95', '71', '226', '313']

                image_filename = strs[0].strip()  # 000001.jpg
                image_file = os.path.join(img_dir, image_filename)  # D:\CelebA\img_celeba\000001.jpg

                with Image.open(image_file) as img:
                    img_w, img_h = img.size  # 得到图片的宽和高

                    # 得到标签框左上角和右下角的坐标值，并转为浮点型
                    x1 = int(strs[1].strip())
                    y1 = int(strs[2].strip())
                    w = int(strs[3].strip())
                    h = int(strs[4].strip())
                    x2 = int(x1 + w)
                    y2 = int(y1 + h)

                    if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue  # 将标签框不符合条件的跳过

                    boxes = [[x1, y1, x2, y2]]  # 拿到标签框的四个坐标值，比如[[95, 71, 321, 384]]

                    # 计算出人脸中心点位置
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    # 生成正样本、部分样本、负样本（很少）
                    for _ in range(200):
                        # 让人脸中心点有少许的偏移
                        w_ = np.random.randint(int(-w * 0.2), int(w * 0.2))
                        h_ = np.random.randint(int(-h * 0.2), int(h * 0.2))
                        cx_ = cx + w_
                        cy_ = cy + h_

                        # 形成正方形边框，并且让坐标也有少许的偏离
                        side_len = np.random.randint(int(min(w, h) * 0.5), np.ceil(0.8 * max(w, h)))

                        x1_ = np.max(cx_ - side_len / 2, 0)
                        y1_ = np.max(cy_ - side_len / 2, 0)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        crop_box = np.array([x1_, y1_, x2_, y2_])  # 裁剪框

                        # 计算坐标的偏移值：标签框的横坐标减去裁剪框的横坐标再除以裁剪框的边长
                        offset_x1 = (x1 - x1_) / side_len
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        # 剪切下图片，并进行大小缩放
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size))

                        # print(utils.iou(crop_box, np.array(boxes)))  # [0.21736549]

                        iou = utils.iou(crop_box, np.array(boxes))[0]  # 裁剪框与标签框做IOU取零轴，得到0.2173654895529984

                        if iou > 0.65:  # 正样本
                            positive_anno_file.write(  # 往正样本txt文件写入信息
                                "positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                    positive_count, 1, offset_x1, offset_y1,
                                    offset_x2, offset_y2))
                            positive_anno_file.flush()  # 释放内存
                            face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            positive_count += 1
                        elif 0.1 < iou < 0.2:  # 部分样本
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                    part_count, 2, offset_x1, offset_y1, offset_x2,
                                    offset_y2))
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            part_count += 1
                        elif iou < 0.05:
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1

                        # 拿到标签框，用新的算法生成更多的负样本
                        _boxes = np.array(boxes)
                    for i in range(200):

                        side_len = np.random.randint(face_size, min(img_w, img_h) / 1.5)
                        x_ = np.random.randint(0, img_w - side_len)
                        y_ = np.random.randint(0, img_h - side_len)
                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                        if np.max(utils.iou(crop_box, _boxes)) < 0.01:  # 裁剪框和标签框作比较
                            face_crop = img.crop(crop_box)
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)  # 防止图像变形

                            negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1

            except Exception as e:
                traceback.print_exc()

    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()
