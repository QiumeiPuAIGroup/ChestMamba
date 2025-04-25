import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable  # Tensor是存在Variable中的.data里的，而cpu和gpu的数据是通过 .cpu()和.cuda()来转换的

import torch.nn as nn
import torch.nn.functional as F
import torch
# from models.gan_myself import Discriminator,Generator
from utils.gan_dataset import get_dataloader_gan

from models.create_gan_model import create_model

# from torchvision.utils import save_image

from utils.common_utils import create_folder


def train(train_dataloader):

    # 模型
    img_shape = (opt.channels, opt.img_size, opt.img_size)

    generator, discriminator = create_model(opt.model_name, img_shape, z_dim=100)
    


    # Optimizers
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    optimizer_G = torch.optim.AdamW(generator.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=50, eta_min=0.00001, last_epoch=-1)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=50, eta_min=0.00001, last_epoch=-1)

    # 是否有cuda
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()
        # generator.load_from()
        # discriminator.load_from()

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    batches_done = 0

    # ----------
    #  Training
    # ----------
    # 进行多个epoch训练
    for epoch in range(opt.n_epochs):
        for i, imgs in enumerate(train_dataloader):  # 这就要看数据是如何dataloader的了
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

            loss_D.backward()
            optimizer_D.step()
            # scheduler_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z)
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_imgs))

                loss_G.backward()
                optimizer_G.step()
                # scheduler_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, batches_done % len(train_dataloader), len(train_dataloader), loss_D.item(), loss_G.item())
                )

            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:1], os.path.join(opt.save_example_path, f'{batches_done}.jpg'), nrow=1, normalize=True)
            batches_done += 1

                ## 保存模型
        torch.save(generator.state_dict(), os.path.join(opt.save_path, 'generator.pth'))
        torch.save(discriminator.state_dict(), os.path.join(opt.save_path, 'discriminator.pth'))


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=2000, help="interval betwen image samples")
    parser.add_argument("--model-name", type=str, default='mamba_gan', help="model name")
    parser.add_argument("--data-path", type=str, default='data/COVID-19_Radiography_Dataset/COVID/images', help="data path")
    parser.add_argument("--save-path", type=str, default='checkpoints_gan/GAN/MambaGan_myself', help="save path")
    parser.add_argument("--save_img_path", type=str, default='checkpoints_gan/GAN/MambaGan_myself/fake_imgs', help="save img path")
    parser.add_argument("--save-testloader-path", type=str, default='checkpoints_gan/GAN/MambaGan_myself/test_imgs', help="save testloader path")
    parser.add_argument("--generator_num", type=int, default=100,help="generate num")
    parser.add_argument("--save_example_path", type=str, default='checkpoints_gan/GAN/MambaGan_myself/example_imgs', help="save example path")
    parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")

    opt = parser.parse_args()
    print(opt)
    return opt


def save_to_eval(test_dataloader, save_path, save_img_path, save_testloader_path, generator_num):
    index = 0
    # 保存测试集图像
    for i, imgs in enumerate(test_dataloader):
        for i in range(imgs.size(0)):
            save_image(imgs[i], os.path.join(save_testloader_path, f'{index}.jpg'))
            index += 1


    # 保存生成图像
    img_shape = (opt.channels, opt.img_size, opt.img_size)
    generator, discriminator = create_model(opt.model_name, img_shape, z_dim=100)

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()
        
    state_dict = torch.load(os.path.join(save_path, 'generator.pth'), map_location='cuda')
    generator.load_state_dict(state_dict)

    generator.eval()
    
    for i in range(generator_num):
        z = Variable(torch.randn(1,opt.latent_dim)).cuda()
        img = generator(z)

        save_image(img[0], os.path.join(save_img_path, f'{i}.jpg'))
    
    


if __name__ == '__main__':
    opt = get_argparse()

    # 生成文件夹
    create_folder(opt.data_path)
    create_folder(opt.save_path)
    create_folder(opt.save_img_path)
    create_folder(opt.save_example_path)
    create_folder(opt.save_testloader_path)

    # 读取数据
    train_dataloader, test_dataloader = get_dataloader_gan(opt.data_path, opt.batch_size,num_workers=2)
    train(train_dataloader)

    save_to_eval(test_dataloader, opt.save_path, opt.save_img_path, opt.save_testloader_path, opt.generator_num)

