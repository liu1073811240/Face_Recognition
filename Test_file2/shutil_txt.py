import shutil
import os

# for line in open(r"G:\list_bbox_celeba2.txt"):
#     print(line)
#     exit()
import glob
import os

# path = r"G:\list_bbox_celeba2.txt"
# f = r"G:\list_bbox_celeba3.txt"
#
# file = open(path)
# for line in file.readlines():
#     print(line)
#     strs = line.strip().split(" ")
#     print(strs)
#     # shutil.copyfile(path + '/' + line, f + '/' + line)
#     shutil.copy(path + "strs", f + "strs")
#     exit()


# _*_ coding:utf-8 _*_
# 开发人员 ： lenovo
# 开发时间 ：2019/12/2920:44
# 文件名称 ：Processing.py
# 开发工具 ： PyCharm

# 对实现收集好的数据进行预处理（将单独的名词，换成句子（定义类））


# _*_ coding:utf-8 _*_

with open(r"D:\CelebA\list_bbox_celeba.txt", "r") as f1:  # 原txt存放路径
    # with open("Noun.txt","r") as f:
    Read_data = f1.readlines()  # 将打开文件的内容读到内存中，with 在执行完命令后，会关闭文件

f2 = open(r"D:\ACelebA\CelebA_5K.txt", "w")  # 新txt存放路径
# 此处如果是'wb',则会出现TypeError: a bytes-like object is required, not 'str'的报错
# 'wb'表示每次写入前格式化文本，如果此文件不存在，则创建一个此文件名的文件
# f2.write("这是一个测试")
count = 0
for x in Read_data:
    f2.write(x)  # 将原记事本的文件写入到另外一个记事本

    count += 1
    if count == 1002:
        break

f2.close()  # 执行完毕，关闭文件
# print(x)#
# print(data,"\n")
