import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import time
import yaml
import tqdm
import logging

import sys
from pathlib import Path
current_folder = Path(__file__).absolute().parent  # ugly
father_folder = str(current_folder.parent)
sys.path.insert(0, father_folder)

from utils import set_seed, value2class, mkdir, inverse_PM
from datasets import ImagePMSet, get_transform
from networks import get_nets
from metric_counter import MetricCounter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 此处需继续优化
with open('networks/config/config.yaml', 'r') as file:
    config_list = yaml.load(file, Loader=yaml.FullLoader)
    data_dir = config_list['nonsky_dir']
    train_fig = config_list['train']
    train_pretrained = train_fig['pretrained']
    train_batch = config_list['batch_size']
    train_imgsize = config_list['image_size']


class Trainer:
    def __init__(self, config, train: DataLoader, val: DataLoader):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.metric_counter = MetricCounter(config['experiment_desc'])

    def train(self):
        self._init_params()
        for epoch in range(0, self.config['num_epochs']):
            self._run_epoch(epoch)
            self._validate(epoch)

            # PSNR bug!
            if self.metric_counter.update_best_model(): 
                torch.save({
                    'model': self.model.state_dict()
                }, 'best_{}.h5'.format(self.config['experiment_desc']))
            torch.save({
                'model': self.model.state_dict()
            }, 'last_{}.h5'.format(self.config['experiment_desc']))

            # loss message bug!
            print(self.metric_counter.loss_message())
            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                self.config['experiment_desc'], epoch, self.metric_counter.loss_message()))

    def _run_epoch(self, epoch):
        self.metric_counter.clear()
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']  # ??

        epoch_size = self.config.get('train_batches_per_epoch') or len(self.train_dataset)
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0
        loss_mean = 0.
        correct = 0.
        total = 0.
        for data in tq:
            # 抽象一层 model 管理 networks
            # inputs, targets = self.model.get_input(data)
            inputs, labels = data
            labels = labels.view(len(labels), -1)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self.model(inputs)
            labels = labels.float()

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total += labels.size(0)
            outputs, labels = map(inverse_PM, (outputs, labels))
            outputs, labels = map(value2class, (outputs, labels))
            correct += torch.sum(outputs == labels).item()

            self.metric_counter.add_losses(loss.item(), loss.item(), 0.1)
            self.metric_counter.add_metrics(correct, correct)
            tq.set_postfix(loss=self.metric_counter.loss_message())
           
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch)

    def _validate(self, epoch):
        self.metric_counter.clear()
        epoch_size = self.config.get('val_batches_per_epoch') or len(self.val_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        i = 0
        loss_mean = 0.
        correct = 0.
        total = 0.
        for data in tq:
            inputs, labels = data
            labels = labels.view(len(labels), -1)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self.model(inputs)
            labels = labels.float()

            loss = self.criterion(outputs, labels)
            total += labels.size(0)
            outputs, labels = map(inverse_PM, (outputs, labels))
            outputs, labels = map(value2class, (outputs, labels))
            correct += torch.sum(outputs == labels).item()

            self.metric_counter.add_losses(loss.item(), loss.item(), 0.1)
            self.metric_counter.add_metrics(correct, correct)

            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch, validation=True)

    def _get_optim(self, params):
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'sgd':
            optimizer = optim.SGD(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'adadelta':
            optimizer = optim.Adadelta(params, lr=self.config['optimizer']['lr'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _init_params(self):
        self.criterion = nn.MSELoss()  # get_loss 抽象
        self.model = get_nets(self.config['model'])
        self.model.to(device)
        self.optimizer = self._get_optim(filter(lambda p: p.requires_grad, self.model.parameters()))


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

    trainer = Trainer(config_list, train=train_loader, val=valid_loader)
    trainer.train()
    

if __name__ == "__main__":
    mainFunc()








