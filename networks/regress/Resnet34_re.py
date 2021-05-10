import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import time
import yaml

import sys
from pathlib import Path
current_folder = Path(__file__).absolute().parent  # ugly
father_folder = str(current_folder.parent)
sys.path.insert(0, father_folder)

# print(sys.path)
# # from . import eval_re
from utils import set_seed, value2class
from datasets import ImagePMSet, get_transform
from models import resnet34_custom


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('networks/config/config.yaml') as file:
    config_list = yaml.load(file, Loader=yaml.FullLoader)
    data_dir = config_list['nonsky_dir']
    train_fig = config_list['train']
    train_epochs = train_fig['epochs']
    train_pretrained = train_fig['pretrained']
    train_batch = train_fig['batch']
    train_imgsize = train_fig['imgsize']

# set parameters 超参数可以根据自己需要调
LR = 0.0005
log_interval = 10   # 每个 epoch 中输出日志间隔
val_interval = 1
classes = 1          # 回归输出一个标量
start_epoch = -1
lr_decay_step = 7


def train(model, data_loader, criterion, optimizer, imagePMSet):
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=lr_decay_step, gamma=0.1)

    # start training
    time_start=time.time()
    train_curve = list()
    valid_curve = list()
    iter_count = 0

    # construct SummaryWriter
    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")
    train_loader = data_loader['train']
    model.to(device)
    for epoch in range(start_epoch + 1, train_epochs):
        if epoch > 1:
            break

        loss_mean = 0.
        correct = 0.
        total = 0.

        model.train()
        for i, data in enumerate(train_loader):

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
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}]" \
                    " Loss: {:.4f} Acc:{:.2%}".format(
                        epoch, train_epochs, i+1, len(train_loader), loss_mean, correct / total))

                loss_mean = 0.

            # log the data, save to "event file"
            writer.add_scalars("Loss", {"Train": loss.item()}, iter_count)
            writer.add_scalars("Accuracy", {"Train": correct / total}, iter_count)

        scheduler.step()  # update learning rate

        # validate the model
        valid_loader = data_loader['val']
        if (epoch+1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.

            model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    labels = labels.view(len(labels), -1)
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    total_val += labels.size(0)
                    outputs, labels = map(imagePMSet.inverse_PM, (outputs, labels))
                    outputs, labels = map(value2class, (outputs, labels))
                    correct_val += torch.sum(outputs == labels).item()

                    loss_val += loss.item()

                loss_val_mean = loss_val / len(valid_loader)  # 除以批次数目
                valid_curve.append(loss_val_mean)
                print("Valid:   Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}]" \
                    " Loss: {:.4f} Acc:{:.2%}".format(
                        epoch, train_epochs, j+1, len(valid_loader), loss_val_mean, correct_val / total_val))

                # log the data, save to "event file"
                writer.add_scalars("Loss", {"Valid": loss_val_mean}, iter_count)
                writer.add_scalars("Accuracy", {"Valid": correct_val / total_val}, iter_count)

    writer.close()
    time_end = time.time()
    print('totally time cost {:.2f} s'.format(time_end - time_start))
    return model


def mainFunc():
    set_seed(1)  # set random seed

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "val")

    train_transform = get_transform(train_imgsize, 'Resize')
    valid_transform = get_transform(train_imgsize, 'Resize')

    # construct dataset, DataLoder
    train_data = ImagePMSet(root=train_dir, transform=train_transform)
    valid_data = ImagePMSet(root=valid_dir, transform=valid_transform)
    train_loader = DataLoader(dataset=train_data, batch_size=train_batch, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=train_batch)
    data_loaders = {'train': train_loader, 'val': valid_loader}

    # construct model
    resnet34_ft = resnet34_custom(classes, pretrained=True)
    criterion = nn.MSELoss() 

    fc_params_id = list(map(id, resnet34_ft.fc.parameters()))
    base_params = filter(lambda p: id(p) not in fc_params_id, resnet34_ft.parameters())
    optimizer = optim.SGD([
        {'params': base_params, 'lr': LR},
        {'params': resnet34_ft.fc.parameters(), 'lr': LR}], 
        momentum=0.9)

    model = train(resnet34_ft, data_loaders, criterion, optimizer, train_data)
    torch.save(model.state_dict(), 'data/Resnet/resnet_re.pt')
    

if __name__ == "__main__":
    mainFunc()








