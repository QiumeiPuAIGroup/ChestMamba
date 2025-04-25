import torch.nn as nn
import torch
import numpy as np

class Discriminator(nn.Module):
    def __init__(self,img_shape):
        super(Discriminator,self).__init__()
        self.img_shape = img_shape # img_shape为单张图片的shape例如（1,256,256）
        # img_area为单张图片的大小，比如（1,256,256）的图像img_area=1*256*256
        self.img_area = np.prod(img_shape)
        self.model = nn.Sequential(
            nn.Linear(self.img_area,512), 
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid(),  # 将二分类问题映射到[0,1]
        )

    def forward(self,img):
        img_flat = img.view(img.size(0),-1)  # 将图像变为序列
        validity = self.model(img_flat)  # 通过鉴别器鉴别
        return validity   # 鉴别器返回的是[0,1]之间的概率
    
# 输入100维的0~1之间的高斯分布，然后通过第一层线性变换将其映射到256维，也就是说输入shape为（batch_size，100），而不是图像的形状
class Generator(nn.Module):
    def __init__(self,img_shape,z_dim=100):
        super(Generator,self).__init__()

        self.z_dim = z_dim  # 输入的这100维的100是可以自己确定的
        self.img_area = np.prod(img_shape)
        self.img_shape = img_shape
        # 模型中间块
        def block(in_feat,out_feat,normalize=True):
            layers = [nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat,0.8)) # 正则化
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(self.z_dim,128,normalize=False),
            *block(128,256),
            *block(256,512),
            *block(512,1024),
            nn.Linear(1024,self.img_area),
            nn.Tanh()  # 将（img_area)的数据映射到[-1,1]之间
        )

    def forward(self, z):
        imgs = self.model(z)   # z为输入的噪声，例如（64,100）之类的，64为batch，100为z_dim
        imgs = imgs.view(imgs.size(0),*self.img_shape)
        return imgs
    

if __name__ == '__main__':
    img_shape = (1,256,256)
    generator = Generator(img_shape=img_shape,z_dim=100)
    discriminator = Discriminator(img_shape=(1,256,256))

    # 二分类交叉熵损失函数
    criterion = torch.nn.BCELoss()

    # # Optimizers
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



