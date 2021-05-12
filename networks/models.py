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
