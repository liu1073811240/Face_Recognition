import nets
import train
import os

if __name__ == '__main__':
    net = nets.ONet()
    if not os.path.join("./param6"):
        os.makedirs("./param6")
    trainer = train.Trainer(net, './param6/o_net.pth', r"D:\CelebA_40w\48", r"D:\CelebA_5k_vali\48")
    trainer.train(0.0005)


# 关键点放O网络训练即可，训练起来比较慢

