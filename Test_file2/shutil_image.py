# -*- coding: utf-8 -*-

import shutil
import os

def copy_img():
    '''
    复制、重命名、粘贴文件
    :return:
    '''
    local_img_name = r'D:\CelebA\img_celeba'
    # 指定要复制的图片路径
    path = r'D:\ACelebA\CelebA_5K'
    # 指定存放图片的目录 objFileName()
    for file in os.listdir(local_img_name):
        print(local_img_name + '/' + file)
        shutil.copy(local_img_name + '/' + file, path + '/' + file)
        if file == "001000.jpg":
            break


    # for i in range(1, 100):
    #     new_obj_name = '00000' + str(i) + '.jpg'
    #     print(new_obj_name)
    #     exit()
    #     shutil.copy(local_img_name + '/' + new_obj_name, path + '/' + new_obj_name)


if __name__ == '__main__':
    copy_img()


