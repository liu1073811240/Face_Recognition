from sklearn.metrics import r2_score
from Rocog_face.face import *
import os
from Rocog_face.Mydataset import MyDataset
from torch.utils import data


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = FaceNet().to(self.device)

    def train(self):
        save_path = "params/1.pth"
        if not os.path.exists("params"):
            os.mkdir("params")
        if os.path.exists(save_path):
            self.net.load_state_dict(torch.load(save_path))
        loss_fn = nn.NLLLoss()
        opt = torch.optim.Adam(self.net.parameters())
        batch_size = 50

        mydataset = MyDataset("face_data")
        validate_set = MyDataset("face_data2")
        dataloader = data.DataLoader(dataset=mydataset, shuffle=True, batch_size=batch_size)
        validate_dataloader = data.DataLoader(dataset=validate_set, shuffle=True, batch_size=50)

        for epochs in range(10000):
            loss = 0
            label_list1 = []
            output_list1 = []
            eval_loss = 0
            eval_acc = 0
            for i, (xs, ys) in enumerate(dataloader):
                xs = xs.to(self.device)
                ys = ys.to(self.device)
                # print(ys.shape)  # torch.Size([100])

                feature, cls = self.net(xs)  # torch.Size([100, 512]), # torch.Size([100, 26])
                loss1 = loss_fn(torch.log(cls), ys)

                predict = torch.argmax(cls, dim=1)
                label_list1.extend(ys.data.cpu().numpy().reshape(-1))  # 方便做r2_score
                output_list1.extend(predict.data.cpu().numpy().reshape(-1))

                opt.zero_grad()
                loss1.backward()
                opt.step()

                predict = torch.argmax(cls, dim=1)

                eval_loss += loss1.item() * batch_size  # 一张图片的损失乘以一批图片为批量损失
                eval_acc += (predict == ys).sum().item()

                if i % 10 == 0:
                    print("epoch:{} i:{} loss:{}".format(epochs, i, loss1.item()))

            torch.save(self.net.state_dict(), save_path)
            print("第{}轮参数保存成功".format(epochs))
            print("train_loss:{}".format(loss1.item()))  # 每轮最后一批次的平均损失
            r2 = r2_score(label_list1, output_list1)
            print("训练集第{}轮, r2_score评估分类精度为：{}".format(epochs, r2))
            # 计算精度
            mean_loss = eval_loss / len(mydataset)  # 在全部训练完以后的总损失除以测试数据的个数
            mean_acc = eval_acc / len(mydataset)  # 在全部训练完以后的预测正确的个数除以总个数
            print("平均损失：{0}, 平均精度：{1}".format(mean_loss, mean_acc))

            with torch.no_grad():
                label_list = []
                output_list = []
                for i, (x, y) in enumerate(validate_dataloader):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    feature2, cls = self.net(x)
                    loss = loss_fn(torch.log(cls), y)
                    predict = torch.argmax(cls, dim=1)

                    label_list.extend(y.data.cpu().numpy().reshape(-1))  # 方便做r2_score
                    output_list.extend(predict.data.cpu().numpy().reshape(-1))

                print("vali_loss:{}".format(loss.item()))  # 每轮最后一批次的平均损失
                # print(label_list)
                # print(output_list)
                r2 = r2_score(label_list, output_list)
                print("验证集第{}轮, r2_score评估分类精度为：{}".format(epochs, r2))


if __name__ == '__main__':
    t = Trainer()
    t.train()
