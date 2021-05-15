import torch.nn as nn
import torchvision.models as models


def resnet34_custom(out, pretrained=True):
    """ 自定义 ResNet 最后一层的输出数目 """
    net = models.resnet34(pretrained)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, out)

    return net


def vgg16_customize(out, pretrained=True):
    """替换 vgg 网络最后一层的输出向量

    Params:
        out: 输出向量个数
    """
    net = models.vgg16_bn(pretrained)
    num_features = net.classifier[6].in_features
    features = list(net.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, out)])
    net.classifier = nn.Sequential(*features)  # Replace the model classifier

    return net


def get_nets(model_config):
    model_name = model_config['g_name']
    out_features = model_config['out_features']

    if 'resnet34' == model_name:
        model = resnet34_custom(out_features)
    elif 'vgg16' == model_name:
        model = vgg16_customize(out_features)

    return model