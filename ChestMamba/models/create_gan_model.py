import torch
from models.gan_myself import Generator as Gan_Generator
from models.gan_myself import Discriminator as Gan_Discriminator
from models.wgan_myself import Generator as WGan_Generator
from models.wgan_myself import Discriminator as WGan_Discriminator
from models.dcgan import Generator as DCGan_Generator
from models.dcgan import Discriminator as DCGan_Discriminator
from models.mamba_gan import Mamba_Generator, Mamba_Discriminator

def create_model(model_name, img_shape, z_dim=100):
    if model_name == 'gan':
        G = Gan_Generator(img_shape, z_dim)
        D = Gan_Discriminator(img_shape)
    elif model_name == 'wgan':
        G = WGan_Generator(img_shape, z_dim)
        D = WGan_Discriminator(img_shape)
    elif model_name == 'dcgan':
        G = DCGan_Generator(z_dim=z_dim, img_shape=img_shape)
        D = DCGan_Discriminator(img_shape)
    elif model_name == 'mamba_gan':
        G = Mamba_Generator(lantent_dim=z_dim)
        D = Mamba_Discriminator(num_classes=1)
    # elif model_name == 'wgan':
    #     G = WGan_Generator()
    #     D = WGan_Discriminator()
    # elif model_name == 'DCGan':
    #     G = DCGan_Generator()
    #     D = DCGan_Discriminator()
    
    return G, D