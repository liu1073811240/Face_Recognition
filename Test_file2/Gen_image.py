import os
import numpy as np
from PIL import Image


def gen_datasets(bg_path, face_image, img_path, label_path, count):
    for _ in range(10):
        with open(label_path, "a") as f:
            # count += 1
            for name in os.listdir(face_image):

                # 随机拿一张背景图片出来
                filename = np.random.randint(1, 1008)
                bg_img = Image.open("{0}/{1}.jpg".format(bg_path, filename))
                bg_img = bg_img.convert("RGB")
                bg_img = bg_img.resize((500, 500))
                # bg_img.save("{0}/{1}.png".format(img_path, count))  # 保存背景图片
                # f.write("{}.png {} {} {} {} {}\n".format(count, 0, 0, 0, 0, 0))  # 在txt文件写入背景图片标签

                face_img1 = Image.open("{}/{}".format(face_image, name))

                new_w = np.random.randint(100, 200)
                new_h = np.random.randint(100, 200)
                resize_img = face_img1.resize((new_w, new_h))  # 随机缩放
                # rot_img = resize_img.rotate(np.random.randint(-45, 45))  # 经过处理后得到的人脸的图片

                paste_x1 = np.random.randint(0, 500-new_w)
                paste_y1 = np.random.randint(0, 500-new_h)

                r, g, b, a = resize_img.split()
                bg_img.paste(resize_img, (paste_x1, paste_y1), mask=a)  # 背景图片上粘贴人脸图片
                paste_x2 = paste_x1 + new_w
                paste_y2 = paste_y1 + new_h
                # print(np.shape(bg_img))

                bg_img.save("{}/{}.jpg".format(img_path, str(count).zfill(6)))  # 保存合成图片
                f.write("{}.jpg {} {} {} {}\n".format(
                    str(count).zfill(6), paste_x1, paste_y1, new_w, new_h))  # 在txt文件写入合成图片标签

                count += 1

                # if count == 5:
                #     print(count)
                #     break

if __name__ == '__main__':
    count = 1
    # 背景图片路径
    bg_img1 = r"D:\ACelebA\Bg_image"
    # bg_img2 = r"D:\PycharmProjects\2020-09-08-minions_reg\Dataset\Bg_Image_train"
    # bg_img3 = r"./Dataset/Bg_Image_test"

    face_img = r"D:\ACelebA\face_image"  # 人脸图片路径

    train_img = r"D:\ACelebA\Gen_image"  # 合成图片
    # validate_img = r"./Dataset/Validate_Data"
    # face_images = r"./Dataset/Test_Data"

    train_label = r"D:\ACelebA\Gen.txt"  # 训练图片标签
    # validate_label = r"./Target/validate_label.txt"
    # test_label = r"./Target/test_label.txt"

    gen_datasets(bg_img1, face_img, train_img, train_label, count)
    # gen_datasets(bg_img2, minions_img, validate_img, validate_label)
    # gen_datasets(bg_img3, minions_img, face_images, test_label)



