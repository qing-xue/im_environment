import os
import matplotlib.pyplot as plt
import torchvision
import torch
import numpy as np
import random

plt.ion()  


def mkdir(floder):
    if not os.path.exists(floder):
        os.makedirs(floder)


# # 先记录下来，后面再写入文件. 42.245478541401894, 15.92258928945192
def inverse_PM(PMs, PM_mean=42.25, PM_std=15.92):
    return PMs * PM_std + PM_mean


def im_segment(input_):
    """ 截取图像中非天空区域 """
    w, h = input_.size
    box = (0, h * 2 // 3, w, h)  # 暂时粗略截
    input_ = input_.crop(box)   
    return input_


def value2class(PMs, pollution={'L0':35, 'L1':70, 'L2':100}):
    classes = np.zeros(len(PMs))
    for i, x in enumerate(PMs):
        if x <= pollution['L0']:
            classes[i] = int(0)
        elif x <= pollution['L1']:
            classes[i] = int(1)
        elif x <= pollution['L2']:
            classes[i] = int(2)
        else:
            classes[i] = int(2)  # 有剩余的也暂时归入最后一类

    return classes


def set_seed(seed):
    """ 需探究这里的 torch, numpy 是否与其他文件一致？ """
    torch.manual_seed(seed)       # cpu 
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True  
    np.random.seed(seed) 
    random.seed(seed)     


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    # plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def show_databatch(inputs, classes, class_names=None):
    if not class_names:
        class_names = classes
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])
