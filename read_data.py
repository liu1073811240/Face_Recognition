from torch.utils.data import DataLoader,Dataset
import os
import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ]
)


class FaceDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())
        # print(self.dataset)  # ['part/21207.jpg 2 -0.5112781954887218 -0.7857142857142857
        # 0.42105263157894735 0.8909774436090225\n'...]
        # print(len(self.dataset))

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split()  # ['negative/87042.jpg', '0', '0', '0', '0', '0']
        img_path = os.path.join(self.path, strs[0])  # G:\48\negative/87042.jpg
        confident = torch.tensor([int(strs[1])],  dtype=torch.float32)  # tensor([0.])
        offset = torch.tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])  # tensor([0., 0., 0., 0.])

        img_data = Image.open(img_path)
        img_data = transform(img_data)
        return img_data, confident, offset

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    train_data = FaceDataset(r"D:\CelebA_5K_vali\12")
    # print(dataset)
    # save_train_file = r"D:\pycharmprojects\MTCNN3\train_data"

    # torch.save(train_data, save_train_file)  # 保存成训练集文件，加载速度更快
    # train_data = torch.load(save_train_file)

    data = DataLoader(dataset=train_data, batch_size=100, shuffle=True, num_workers=4)
    for i, (img, con, off) in enumerate(data):
        print(img.shape)
        print(con.shape)
        print(off.shape)



