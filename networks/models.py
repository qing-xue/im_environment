import torch
from torch.autograd import Variable
import time
import utils
import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def visualize_model(vgg, dataloader, num_images=6, class_names=None):
    """预测结果可视化

    参数说明：
        dataloaders: 载入测试数据
    """
    was_training = vgg.training
    
    # Set model for evaluation
    vgg.train(False)
    vgg.eval() 
    
    images_so_far = 0

    for i, data in enumerate(dataloader):
        inputs, labels = data
        size = inputs.size()[0]

        inputs, labels = Variable(inputs.to(device), volatile=True), Variable(labels.to(device), volatile=True)
        
        outputs = vgg(inputs)
        
        _, preds = torch.max(outputs.data, 1)
        predicted_labels = [preds[j] for j in range(inputs.size()[0])]
        
        print("Ground truth:")
        utils.show_databatch(inputs.data.cpu(), labels.data.cpu(), class_names)
        print("Prediction:")
        utils.show_databatch(inputs.data.cpu(), predicted_labels, class_names)
        
        del inputs, labels, outputs, preds, predicted_labels
        torch.cuda.empty_cache()
        
        images_so_far += size
        if images_so_far >= num_images:
            break
        
    vgg.train(mode=was_training)  # Revert model back to original training state


def vgg16_customize(vgg16, out):
    """替换 vgg 网络最后一层的输出向量

    Params:
        out: 输出向量个数
    """
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, out)])
    vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier

    print(vgg16)
    return vgg16


def resnet34_custom(out, pretrained=True):
    """ 自定义 ResNet 最后一层的输出数目 """
    net = models.resnet34(pretrained)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, out)

    return net


if __name__ == '__main__':
    net = resnet34_custom(3)
    print(net)
