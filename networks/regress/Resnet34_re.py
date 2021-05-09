import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import time

import sys
from pathlib import Path
current_folder = Path(__file__).absolute().parent  # ugly
father_folder = str(current_folder.parent)
sys.path.append(father_folder)

from utils import set_seed, value2class
from datasets import ImagePMSet
from models import resnet34_custom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set parameters 超参数可以根据自己需要调
MAX_EPOCH = 10
BATCH_SIZE = 16
LR = 0.0005
log_interval = 10
val_interval = 1
classes = 1  # 回归输出一个标量
start_epoch = -1
lr_decay_step = 7


def train(model, data_loader, criterion, optimizer, imagePMSet):
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=lr_decay_step, gamma=0.1)

    # start training
    time_start=time.time()
    train_curve = list()
    iter_count = 0

    # construct SummaryWriter
    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    for epoch in range(start_epoch + 1, MAX_EPOCH):
        loss_mean = 0.
        correct = 0.
        total = 0.

        model.train()
        for i, data in enumerate(data_loader):

            iter_count += 1

            # forward
            inputs, labels = data
            labels = labels.view(len(labels), -1)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            labels = labels.float()

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # count class distribution 此处需改进！
            total += labels.size(0)
            outputs, labels = map(imagePMSet.inverse_PM, (outputs, labels))
            outputs, labels = map(value2class, (outputs, labels))
            correct += torch.sum(outputs == labels).item()

            # print the training information
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] \
                    Loss: {:.4f} \ Acc:{:.2%}".format(
                        epoch, MAX_EPOCH, i+1, len(data_loader), 
                        loss_mean, correct / total))
                loss_mean = 0.

            # log the data, save to "event file"
            writer.add_scalars("Loss", {"Train": loss.item()}, iter_count)
            writer.add_scalars("Accuracy", {"Train": correct / total}, iter_count)

        scheduler.step()  # update learning rate

    time_end=time.time()
    print('totally cost', time_end-time_start)


def mainFunc(data_dir, imgsize):
    """
    函数入口参数：
        data_dir 图片数据集
        imgsize 输入图片尺寸（Eg:图片尺寸为128*128时，imgsize=128）
    """
    if not os.path.exists(data_dir):
        raise Exception("\n{} not exists".format(data_dir))

    set_seed(1)  # set random seed

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "val")

    norm_mean = [0.485, 0.456, 0.406]  # from Imagenet
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(imgsize),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(imgsize),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # construct dataset, DataLoder
    train_data = ImagePMSet(root=train_dir, transform=train_transform)
    valid_data = ImagePMSet(root=valid_dir, transform=valid_transform)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

    # construct model
    resnet34_ft = resnet34_custom(classes, pretrained=True)
    resnet34_ft.to(device)
    criterion = nn.MSELoss() 

    fc_params_id = list(map(id, resnet34_ft.fc.parameters()))
    base_params = filter(lambda p: id(p) not in fc_params_id, resnet34_ft.parameters())
    optimizer = optim.SGD([
        {'params': base_params, 'lr': LR},
        {'params': resnet34_ft.fc.parameters(), 'lr': LR}], 
        momentum=0.9)

    train(resnet34_ft, train_loader, criterion, optimizer, train_data)
    

if __name__ == "__main__":
    data_dir = r'F:\workplace\public_dataset\Heshan_imgset\256x256\non_sky'
    mainFunc(data_dir, imgsize=128)








