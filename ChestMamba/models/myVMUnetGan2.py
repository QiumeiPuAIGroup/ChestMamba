from models.vmunet.vmamba import VSSM
from torch import nn
import torch
from models.vmunet.MyVMamba import VSSM_D

class VMUNetG(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=3,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2,9,2,2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           )
        
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.vmunet(x)
        if self.num_classes == 1: return torch.sigmoid(logits)
        else: return logits
    
    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少参数
            print('Total model_dict:{},Total pretrained_dict:{},update:{}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layer.0' in k:
                    new_k = k.replace('layer.0', 'layer_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k: 
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k: 
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k: 
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v

             # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)
            
            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")


class Discriminator(nn.Module):
    def __init__(self,
                 input_channels=6,
                 num_classes=1,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2,9,2,2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM_D(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           )


    def forward(self, img_A, img_B):
        x = torch.cat((img_A, img_B), 1)
        logits = self.vmunet(x)
        if self.num_classes == 3: return torch.sigmoid(logits)
        else: return logits

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
if __name__ == '__main__':
    model = VMUNetG(num_classes=3).cuda()
    model_D = Discriminator().cuda()
    print(model)
    input_img = torch.randn(4,3,256,256).cuda()
    img_A = model(input_img)
    print(img_A.shape)
    img_B = torch.randn(4,3,256,256).cuda()
    out = model_D(img_A, img_B)
    print(out.shape)
