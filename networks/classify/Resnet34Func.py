import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import time
# from tools.common_tools import set_seed
import sys
from datasets import ImgDataset
from models import resnet34_custom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set parameters 超参数可以根据自己需要调
MAX_EPOCH = 100
BATCH_SIZE = 16
LR = 0.0005
log_interval = 10
val_interval = 1
classes = 3
start_epoch = -1
lr_decay_step = 7


def train(model, data_loader, criterion, optimizer):
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
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # count class distribution
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().cpu().sum().numpy()

            # print the training information
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i+1, len(data_loader), loss_mean, correct / total))
                loss_mean = 0.

            # log the data, save to "event file"
            writer.add_scalars("Loss", {"Train": loss.item()}, iter_count)
            writer.add_scalars("Accuracy", {"Train": correct / total}, iter_count)

        scheduler.step()  # update learning rate

    time_end=time.time()
    print('totally cost',time_end-time_start)


def mainFunc(data_dir, path_pretrained_model, imgsize):
    """
    函数入口参数：
        data_dir 图片数据集
        path_pretrained_model 预训练模型路径
        imgsize 输入图片尺寸（Eg:图片尺寸为128*128时，imgsize=128）
    """
    DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."+os.path.sep+"..")
    sys.path.append(DIR)

    BASEDIR = os.path.dirname(os.path.abspath(__file__))
    print("use device :{}".format(device))

    # set_seed(1)  # set random seed
    label_name = {"0": 0, "1": 1,"2" : 2}

    # data_input
    # data_dir = os.path.abspath(os.path.join(r"D:\BaiduNetdiskDownload\Proenviroment\Heshanimgset\Heshanimgset\Results", "split"))
    if not os.path.exists(data_dir):
        raise Exception("\n{} not exists, please put to \n{} ".format(
            data_dir, os.path.dirname(data_dir)))

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "val")

    print(train_dir)
    print(valid_dir)

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(imgsize),
        transforms.CenterCrop(imgsize - 6),
        transforms.RandomResizedCrop(imgsize - 8),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(imgsize),
        transforms.CenterCrop(imgsize - 8),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # construct dataset
    train_data = ImgDataset(data_dir=train_dir, transform=train_transform)
    valid_data = ImgDataset(data_dir=valid_dir, transform=valid_transform)

    # construct DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

    # construct model
    resnet34_ft = resnet34_custom(classes, pretrained=True)

    # freeze the conv
    flag_m1 = 0
    if flag_m1:
        for param in resnet34_ft.parameters():
            param.requires_grad = False
        print("conv1.weights[0, 0, ...]:\n {}".format(resnet34_ft.conv1.weight[0, 0, ...]))

    resnet34_ft.to(device)
    # loss function
    criterion = nn.CrossEntropyLoss()  # choose loss function

    fc_params_id = list(map(id, resnet34_ft.fc.parameters()))
    base_params = filter(lambda p: id(p) not in fc_params_id, resnet34_ft.parameters())
    optimizer = optim.SGD([
        {'params': base_params, 'lr': LR},
        {'params': resnet34_ft.fc.parameters(), 'lr': LR}], momentum=0.9)

    train(resnet34_ft, train_loader, criterion, optimizer)
    

if __name__ == "__main__":
    # data_dir = 'F:\workplace\public_dataset\Heshan_imgset\256x256\non_sky'
    mainFunc(
        data_dir=r"F:\workplace\public_dataset\Heshan_imgset\256x256\non_sky",
        path_pretrained_model=r".\networks\Resnet\resnet34-333f7ec4.pth",
        imgsize=128)








