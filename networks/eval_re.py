from __future__ import print_function, division
import time
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import yaml
from models import vgg_customize
from datasets import ImagePMSet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def eval_model_re(vgg, criterion, dataloader, imagePMSet, value2class):
    """在测试集上评估模型

    参数说明：
        dataloaders: 载入测试数据
        imagePMSet: 用于逆数据归一化. ！这里引入得不太好
        value2class(): 用于将值数据转为分类等级. ！这里引入得不太好
    """
    since = time.time()
    loss_test = 0
    acc_test = 0

    test_batches = len(dataloader)
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(dataloader):
        if i % 10 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        vgg.train(False)
        vgg.eval()
        inputs, labels = data

        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)

        outputs = vgg(inputs)
        loss = criterion(outputs, labels)
        loss_test += loss.data  # data[0] for GPU?

        outputs, labels = map(imagePMSet.inverse_PM, (outputs, labels))
        outputs, labels = map(value2class, (outputs, labels))
        acc_test += torch.sum(outputs == labels)

        del inputs, labels, outputs
        torch.cuda.empty_cache()

    avg_loss = loss_test / test_batches
    avg_acc = acc_test / test_batches
    elapsed_time = time.time() - since
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)


def value2class(PMs, pollution={'L0':35, 'L1':70, 'L2':100}):
    for i, x in enumerate(PMs):
        if x <= pollution['L0']:
            PMs[i] = pollution['L0']
        elif x <= pollution['L1']:
            PMs[i] = pollution['L1']
        elif x <= pollution['L2']:
            PMs[i] = pollution['L2']
        else:
            PMs[i] = pollution['L2']  # 有剩余的也暂时归入最后一类

    return PMs


with open(r'config\config.yaml') as file:
    config_list = yaml.load(file, Loader=yaml.FullLoader)
    data_dir = config_list['nonsky_dir']
    train_fig = config_list['train']
    train_epochs = train_fig['epochs']

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
vgg16 = models.vgg16_bn()
vgg16 = vgg_customize(vgg16, 1)
vgg16.load_state_dict(torch.load('VGG16/VGG16_20_a1_retrain.pt', map_location=device))

criterion = nn.MSELoss()
eval_model_re(vgg16, criterion, dataloaders['val'], image_datasets['val'], value2class)