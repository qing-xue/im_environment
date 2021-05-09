from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
from models import train_model
import yaml

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('config/config.yaml') as file:
    config_list = yaml.load(file, Loader=yaml.FullLoader)
    data_dir = config_list['nonsky_dir']
    train_fig = config_list['train']
    train_epochs = train_fig['epochs']

# 训练测试字典 key
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
    x: datasets.ImageFolder(
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

print("Classes: ")
class_names = image_datasets[TRAIN].classes
print(image_datasets[TRAIN].classes)

# Load the pretrained model from pytorch
# vgg16 = models.vgg16_bn()
vgg16 = models.vgg16_bn(pretrained=False)  # download 528M
# vgg16.load_state_dict(torch.load("VGG16/vgg16_bn.pth"))

# for param in vgg16.features.parameters():
#     param.require_grad = False  # Freeze training for all layers

# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1]  # Remove last layer
features.extend([nn.Linear(num_features, len(class_names))])  # Add our layer with 3 outputs
vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
print(vgg16.classifier[6])

vgg16.to(device)  # .cuda() will move everything to the GPU side
    
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

vgg16 = train_model(dataloaders, vgg16, criterion, optimizer_ft, num_epochs=train_epochs)
torch.save(vgg16.state_dict(), 'VGG16/VGG16_dataset.pt')

