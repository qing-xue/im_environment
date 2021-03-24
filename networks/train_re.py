from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from datasets import ImagePMSet
from models import vgg_customize
import os
import yaml
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_model_re(dataloader, vgg, criterion, optimizer, num_epochs=10):
    """训练回归模型

    参数说明：
        dataloaders: 载入训练数据. 后期可加验证集
        vgg: 预训练模型
        criterion: 计算损失
        optimizer: 优化器
    """
    since = time.time()
    train_batches = len(dataloader)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        loss_train = 0
        vgg.train(True)

        for i, data in enumerate(dataloader):
            if i % 10 == 0:
                print("\rTraining batch {}/{}\n".format(i, train_batches), end='', flush=True)

            inputs, labels = data
            labels = labels.view(len(labels), -1)
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = vgg(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_train += loss.data  # data[0] for GPU?

            del inputs, labels, outputs
            torch.cuda.empty_cache()

        avg_loss = loss_train / train_batches
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print('-' * 10)

    elapsed_time = time.time() - since
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    return vgg


with open('config/config.yaml') as file:
    config_list = yaml.load(file, Loader=yaml.FullLoader)
    data_dir = config_list['nonsky_re_dir']
    train_fig = config_list['train']
    train_epochs = train_fig['epochs']
    train_pretrained = train_fig['pretrained']

# 训练测试字典 key. 与目录结构相关
TRAIN = 'train'
VAL = 'val'

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transforms = {
    TRAIN: transforms.Compose([
        # Data augmentation 体现在 DataLoader 不同的训练轮次中？
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
}
image_datasets = {
    x: ImagePMSet(
        os.path.join(data_dir, x), 
        transform=data_transforms[x]
    )
    for x in [TRAIN, VAL]
}
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=8,
        shuffle=True, num_workers=0  # 单个线程
    )
    for x in [TRAIN, VAL]
}

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL]}
for x in [TRAIN, VAL]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))

# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn(pretrained=train_pretrained)  # download 528M
if train_pretrained:
    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.require_grad = False

vgg16 = vgg_customize(vgg16, 1)
vgg16.to(device)  # .cuda() will move everything to the GPU side
    
criterion = nn.MSELoss()  # 回归问题改用均方误差
optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

vgg16 = train_model_re(dataloaders['train'], vgg16, criterion, optimizer_ft, num_epochs=train_epochs)
torch.save(vgg16.state_dict(), 'VGG16/VGG16_dataset.pt')

