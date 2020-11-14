from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import numpy

def trans_square(image):
    r"""Open the image using PIL."""
    image = image.convert('RGB')
    w, h = image.size
    background = Image.new('RGB', size=(max(w, h), max(w, h)), color=(0, 0, 0))  # 创建背景图，颜色值为127
    length = int(abs(w - h) // 2)  # 一侧需要填充的长度
    box = (length, 0) if w < h else (0, length)  # 粘贴的位置
    background.paste(image, box)
    # background.save("./merge2.jpg")

    return background


def npz2list(path):
    # 保存列表
    # numpy.savez('list1',list_in)
    list1 = np.load(path, allow_pickle=True)
    # print(list1.files) # 查看各个数组名称
    arr_0 = list1['arr_0']  # object 只读
    list_o = []
    for i in arr_0:
        list_o.append(i)
    return list_o


def cv2ImgAddText(img, text, left, top, textColor=(0, 0, 255), textSize=20):
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    image = Image.open(r"./2.jpg")
    trans_square(image)
