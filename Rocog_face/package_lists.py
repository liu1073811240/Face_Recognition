from PIL import Image
import os
from torchvision import transforms
from Rocog_face.utils import trans_square
from Rocog_face.face import FaceNet
import torch

main_dir = r"./Contrast_data"
tf = transforms.Compose([
    transforms.Resize([112, 112]),  # 不会成比例缩放，图片特征会变形，所以输入图片一定要先转成正方形（缺少的区域填充黑色）
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = r"D:\PycharmProjects\MTCNN_data\Rocog_face\params\1.pth"
net = FaceNet().to(device)
net.load_state_dict(torch.load(save_path))

lists = []
for face_dir in os.listdir(main_dir):
    for face_filename in os.listdir(os.path.join(main_dir, face_dir)):
        img = Image.open(os.path.join(main_dir, face_dir, face_filename))
        img = trans_square(img)

        # 将拿到的图片转成正方形112*112
        person1 = tf(img).to(device)
        print(person1.shape)
        person1_feature = net.encode(torch.unsqueeze(person1, 0))
        print(person1_feature.shape)
        exit()

        lists.extend(person1_feature)

print(lists)


