from PIL import Image
import numpy as np

path = r"C:\Users\lieweiai\Pictures\Camera Roll\2.png"
image = Image.open(path)
img = image.convert("RGB")
print(np.shape(img))
# img_crop = image.crop((100, 100, 200, 200))
#
# img_crop.save(r"C:\Users\lieweiai\Pictures\Camera Roll\1.jpg")

# import cv2
# import numpy
#
# cap = cv2.VideoCapture(0) # 调整参数实现读取视频或调用摄像头
# while 1:
#     ret, frame = cap.read()
#     cv2.imshow("cap", frame)
#     if cv2.waitKey(100) & 0xff == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

