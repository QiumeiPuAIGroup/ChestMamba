import torch.utils.model_zoo as model_zoo
import torch

from models.alexnet import AlexNet
from models.vgg import VGG
from models.mobilenetv2 import MobileNetV2
from models.resnet import ResNet, BasicBlock, Bottleneck
from models.seresnet import BasicResidualSEBlock, SEResNet

from models.model_urls import vgg_model_urls, resnet_model_urls
from models.vit import vit_base_patch16_224, vit_base_patch32_224
from models.swin_transformer import swin_t, swin_s, swin_b, swin_l
from models.inceptionv3 import InceptionV3
from models.mamba_classify import Mamba_classify


def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=3).cuda()
    if pretrained:
        # pretrained_state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
        pretrained_state_dict = torch.load('./pretrained/new_resnet.pth')
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict,strict=False)

    return model


def resnet18_se():
    return SEResNet(BasicResidualSEBlock, [2, 2, 2, 2]).cuda()


def resnet(pretrained=False, depth=18, num_classes=3, **kwargs):
    """Constructs a Resnet model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = None
    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model = model.cuda()

    if pretrained:
        pretrained_state_dict = None
        if depth == 18:
            pretrained_state_dict = model_zoo.load_url(resnet_model_urls['resnet18'])
        elif depth == 34:
            pretrained_state_dict = model_zoo.load_url(resnet_model_urls['resnet34'])
        elif depth == 50:
            pretrained_state_dict = model_zoo.load_url(resnet_model_urls['resnet50'])

        pretrained_state_dict.update(pretrained_state_dict)
        model.load_state_dict(pretrained_state_dict)

    return model


# def alexnet(pretrained=False, **kwargs):
#     """Constructs a AlexNet model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = AlexNet(num_classes=2).cuda()
#     if pretrained:
#         pretrained_state_dict = model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
#         pretrained_state_dict.update(pretrained_state_dict)
#         model.load_state_dict(pretrained_state_dict)

#     return model


def vgg(pretrained=False, depth=16, **kwargs):
    """Constructs a VGG model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param depth: layers of vgg net, such as vgg-16、vgg-19
    """
    model = VGG(depth=depth, num_classes=3).cuda()
    if pretrained:

        pretrained_state_dict = None
        if depth == 16:
            pretrained_state_dict = model_zoo.load_url(vgg_model_urls['vgg16'])
        elif depth == 19:
            pretrained_state_dict = model_zoo.load_url(vgg_model_urls['vgg19'])

        pretrained_state_dict.update(pretrained_state_dict)
        model.load_state_dict(pretrained_state_dict)

    return model


def mobilenetv2(num_classes=2, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = MobileNetV2(num_classes=num_classes).cuda()
    return


def densenet(pretrained=False, depth=121, num_classes=2, **kwargs):
    """Constructs a DenseNet model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    from models.densenet import DenseNet, Bottleneck
    model = None
    if depth == 121:
        model = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_class=num_classes)
    elif depth == 169:
        model = DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32, num_class=num_classes)
    elif depth == 201:
        model = DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32, num_class=num_classes)
    elif depth == 161:
        model = DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48, num_class=num_classes)
    model = model.cuda()

    if pretrained:
        pretrained_state_dict = model_zoo.load_url(vgg_model_urls['vgg16'])
        pretrained_state_dict.update(pretrained_state_dict)
        model.load_state_dict(pretrained_state_dict)

    return model


def create_model(model_name, pretrained=False):
    model = None
    if model_name == 'resnet18-sam':
        model = resnet18_cbam(pretrained)
    elif model_name == 'resnet-se':
        model = resnet18_se()
    elif model_name == 'resnet18':
        model = resnet(pretrained, depth=18)
    elif model_name == 'resnet34':
        model = resnet(pretrained, depth=34)
    elif model_name == 'resnet50':
        model = resnet(pretrained, depth=50)
    elif model_name == 'alexnet':
        model = alexnet(pretrained)
    elif model_name == 'vgg16':
        model = vgg(pretrained, depth=16)
    elif model_name == 'vgg19':
        model = vgg(pretrained, depth=19)
    elif model_name == 'mobilenetv2':
        model = MobileNetV2(num_classes=2).cuda()
    elif model_name == 'vit_patch16':
        model = vit_base_patch16_224(img_size=256, num_classes=4)
    elif model_name == 'vit_patch32':
        model = vit_base_patch32_224(img_size=256, num_classes=4)
    elif model_name == 'swin_t':
        model = swin_t(patch_size=16)
    elif model_name == 'swin_s':
        model = swin_s(patch_size=16)
    elif model_name == 'swin_b':
        model = swin_b(patch_size=16)
    elif model_name == 'swin_l':
        model = swin_l(patch_size=16)
    elif model_name == 'inceptionv3':
        model = InceptionV3(num_classes=3)
    elif model_name == 'mamba_classify':
        model = Mamba_classify(num_classes=4)
    '''
    elif:
    '''

    return model


if __name__ == '__main__':
    name = 'vgg16'
    net = create_model(name, False).cuda()
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    from thop import profile
    flops, params = profile(net, inputs=(dummy_input,), verbose=False)
    print(f"  - FLOPs: {flops / 1e9:.2f} GFlops (billion floating point operations)")
    print(f"  - Parameters: {params / 1e6:.2f} M (million parameters)")

    # '''
    # # Calculate the parameters and computational complexity of the pruned model
    # from nni.compression.pytorch.utils import count_flops_params

    # flops, params, _ = count_flops_params(net, dummy_input, verbose=False)
    # print(f"\nPruned Model after Weight Replacing:\nFLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")
    # '''
