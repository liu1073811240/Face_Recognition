import nets
import train
import os

if __name__ == '__main__':
    net = nets.PNet()
    if not os.path.exists("./param6"):
        os.makedirs("./param6")
    trainer = train.Trainer(net, './param6/p_net.pth', r"D:\CelebA_40w\12", r"D:\CelebA_5K_vali\12")
    trainer.train(0.008)  # 0.01

# 已训练12小时



