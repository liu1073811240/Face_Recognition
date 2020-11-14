import os
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
from read_data import FaceDataset
from tensorboardX import SummaryWriter


# writer = SummaryWriter("./logs")

class Trainer:
    def __init__(self, net, save_path, dataset_path, validate_path, isCuda=True):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.net = net.to(self.device)  # 局部变量实例化，在其它函数也可以调用。
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.validate_path = validate_path

        self.cls_loss_fn = nn.BCELoss()
        self.offset_loss_fn = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters())
        # 接着训练
        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))
            # model = torch.load(self.save_path)

        else:
            print("NO Param")

    def train(self, stop_value):
        batch_size1 = 1024
        batch_size2 = 128
        facedataset1 = FaceDataset(self.dataset_path)  # 训练集
        facedataset2 = FaceDataset(self.validate_path)  # 验证集
        # facedataset1 = torch.load(r"D:\pycharmprojects\MTCNN3\train_data")

        dataloader1 = DataLoader(facedataset1, batch_size=1024, shuffle=True, num_workers=8)  # 1440, 8
        dataloader2 = DataLoader(facedataset2, batch_size=128, shuffle=True, num_workers=2)
        loss_train = 0

        self.net.train()
        # while True:
        for epoch in range(100000):
            # print('123')
            train_loss = 0
            label_category1 = []
            output_category1 = []
            label_offset2 = []
            output_offset2 = []
            for i, (img_data_, category_, offset_) in enumerate(dataloader1):
                img_data_ = img_data_.to(self.device)
                category_ = category_.to(self.device)
                offset_ = offset_.to(self.device)

                _output_category, _output_offset = self.net(img_data_)
                # print(_output_category.shape)  # torch.Size([256, 1, 1, 1])
                # print(_output_offset.shape)  # torch.Size([256, 4, 1, 1])

                output_category = _output_category.reshape(-1, 1)
                # print(output_category.shape)  # torch.Size([256, 1])
                output_offset = _output_offset.reshape(-1, 4)  # torch.Size([256, 4])

                # output_landmark = _output_landmark.view(-1, 10)
                # 计算分类的损失
                category_mask = torch.lt(category_, 2)  # part样本不参与分类损失计算, 输出正样本、负样本的布尔值
                category = torch.masked_select(category_, category_mask)  # 标签：根据掩码选择标签的正样本、负样本，如[1., 0., 1., 0., 1..]
                output_category = torch.masked_select(output_category, category_mask)  # 输出：根据掩码选择输出的正样本、负样本，如[1.0000e+00, 5.5516e-08, 1.0000e+00,.]

                cls_loss = self.cls_loss_fn(output_category, category)

                # 计算bound的损失
                offset_mask = torch.gt(category_, 0)  # 负样本不参与计算， 输出正样本、部分样本的布尔值
                offset = torch.masked_select(offset_, offset_mask)  # 标签：根据掩码选择标签的正样本、部分样本，如[1., 0., 1., 0., 1.。]
                output_offset = torch.masked_select(output_offset, offset_mask)  # 输出：根据掩码选择输出的正样本、部分样本， 如[1.0000e+00, 5.5516e-08, 1.0000e+00,.]

                offset_loss = self.offset_loss_fn(output_offset, offset)  # 损失

                # loss = offset_loss
                loss = 0.5*cls_loss + 0.5*offset_loss  # 两个损失各自优化。谁也不影响谁
                train_loss += loss.item() * batch_size1

                # 评估训练指标
                label_category1.extend(category.data.cpu().numpy().reshape(-1))
                output_category1.extend(output_category.data.cpu().numpy().reshape(-1))
                label_offset2.extend(offset.data.cpu().numpy().reshape(-1))
                output_offset2.extend(output_offset.data.cpu().numpy().reshape(-1))

                # label_category1 = np.stack(label_category1).reshape(-1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cls_loss = cls_loss.cpu().item()
                offset_loss = offset_loss.cpu().item()
                loss_train = loss.cpu().item()
                if i % 50 == 0:

                    print("epoch:", epoch,  "i:", i, "loss:", loss_train,  "cls_loss:", cls_loss, " offset_loss",
                          offset_loss)

            torch.save(self.net.state_dict(), self.save_path)  # 保存参数
            # torch.save(self.model, self.save_path)
            print("params save success")

            # 模型评估：加mAP
            r2 = r2_score(label_category1, output_category1)  # 达标0.95
            print("训练集第{}轮：r2_score评估置信度为：".format(epoch), r2)
            _r2 = r2_score(label_offset2, output_offset2)
            print("训练集第{}轮：r2_score评估偏移量为：".format(epoch), _r2)

        # 加验证集、r2_score评估.比较这一次分数和上一次分数。保存分数高的参数。
            validate_loss = 0
            label_list_category = []
            output_list_category = []
            label_list_offset = []
            output_list_offset = []
            for i, (img_data_, category_, offset_) in enumerate(dataloader2):
                img_data_ = img_data_.to(self.device)
                category_ = category_.to(self.device)
                offset_ = offset_.to(self.device)
                # print(category_.shape)  # torch.Size([1440, 1])
                # print(offset_.shape)  # torch.Size([1440, 4])

                _output_category, _output_offset = self.net(img_data_)
                # print(_output_category.shape)  # torch.Size([256, 1, 1, 1])
                # print(_output_offset.shape)  # torch.Size([256, 4, 1, 1])

                output_category = _output_category.reshape(-1, 1)
                # print(output_category.shape)  # torch.Size([256, 1])
                output_offset = _output_offset.reshape(-1, 4)  # torch.Size([256, 4])

                # output_landmark = _output_landmark.view(-1, 10)

                # 计算分类的损失
                category_mask = torch.lt(category_, 2)  # part样本不参与分类损失计算, 输出正样本、负样本的布尔值
                category = torch.masked_select(category_, category_mask)
                output_category = torch.masked_select(output_category,category_mask)

                cls_loss = self.cls_loss_fn(output_category, category)

                # 计算bound的损失
                offset_mask = torch.gt(category_, 0)  # 负样本不参与计算， 输出正样本、部分样本的布尔值
                offset = torch.masked_select(offset_, offset_mask)  # 标签：根据掩码选择标签的正样本、部分样本，如[1., 0., 1., 0., 1.。]
                output_offset = torch.masked_select(output_offset, offset_mask)  # 输出：根据掩码选择输出的正样本、部分样本， 如[1.0000e+00, 5.5516e-08, 1.0000e+00,.]

                offset_loss = self.offset_loss_fn(output_offset, offset)  # 损失

                # loss = offset_loss
                loss = 0.5*cls_loss + 0.5*offset_loss
                validate_loss += loss.item() * batch_size2

                # 方便做score评估
                label_list_category.extend(category.data.cpu().numpy().reshape(-1))  # 里面包含梯度
                # print(label_list_category)

                output_list_category.extend(output_category.data.cpu().numpy().reshape(-1))
                # print(output_list_category)

                label_list_offset.extend(offset.data.cpu().numpy().reshape(-1))
                # print(label_list_offset)
                output_list_offset.extend(output_offset.data.cpu().numpy().reshape(-1))
                # print(output_list_offset)
                # print("已经枚举到第{}批".format(i+1))

            # 评估坐标值和置信度
            r2 = r2_score(label_list_category, output_list_category)
            print("验证集第{}轮：r2_score评估置信度为：".format(epoch), r2)
            _r2 = r2_score(label_list_offset, output_list_offset)
            print("验证集第{}轮：r2_score评估偏移量为：".format(epoch), _r2)

            mean_train_loss = train_loss / len(facedataset1)
            mean_validate_loss = validate_loss / len(facedataset2)
            print("该轮次的训练平均损失：{}, 验证平均损失：{}".format(mean_train_loss, mean_validate_loss))
            # #
            # writer.add_scalars("loss", {"mean_train_loss": mean_train_loss, "validate_loss": mean_validate_loss}, epoch)
            # tensorboard --logdir=D:\pycharmprojects\MTCNN3\logs --host=127.0.0.1

            if loss_train < stop_value:
                break

# 训练速度一般跟网络模型参数、总数据量大小（包含一张图片、批次）、设备有关


