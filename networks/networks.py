import torch.nn as nn
import torchvision.models as models


def resnet34_custom(out, pretrained=True):
    """ 自定义 ResNet 最后一层的输出数目 """
    net = models.resnet34(pretrained)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, out)

    return net


def get_nets(model_config):
    model_name = model_config['g_name']
    out_features = model_config['out_features']

    if 'resnet34' == model_name:
        model = resnet34_custom(out_features)

    return model