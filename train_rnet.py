import nets
import train
import os

if __name__ == '__main__':
    net = nets.RNet()
    if not os.path.exists("./param6"):
        os.makedirs("./param6")
    trainer = train.Trainer(net, './param6/r_net.pth', r"D:\CelebA_40w\24", r"D:\CelebA_5k_vali\24")
    trainer.train(0.001)

