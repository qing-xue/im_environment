import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
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
set_seed(1)  # set different random seed

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
            self._run_epoch(epoch, valid=True)

            if self.metric_counter.update_best_model(): 
                torch.save({
                    'model': self.model.state_dict()
                }, 'data/best_{}.h5'.format(self.config['experiment_desc']))
            torch.save({
                'model': self.model.state_dict()
            }, 'data/last_{}.h5'.format(self.config['experiment_desc']))

            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                self.config['experiment_desc'], epoch, self.metric_counter.loss_message()))

    def _run_epoch(self, epoch, valid=False):
        self.metric_counter.clear()
        run_dataset = self.train_dataset if not valid else self.val_dataset
        epoch_size = min(self.config.get('batches_per_epoch'), len(run_dataset))
        tq = tqdm.tqdm(run_dataset, total=epoch_size)
        tq.set_description('Epoch {}'.format(epoch) if not valid else "Validation")
        i = 0
        for data in tq:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self.model(inputs)
            
            if not valid:
                self.optimizer.zero_grad()
            loss = self.criterion(outputs, labels)
            if not valid:
                loss.backward()
                self.optimizer.step()

            outputs, labels = map(inverse_PM, (outputs, labels))  # bug
            outputs, labels = map(value2class, (outputs, labels))
            correct = torch.sum(outputs == labels).item()
            acc = correct / labels.size(0)

            self.metric_counter.add_losses(loss.item())
            self.metric_counter.add_metrics(acc)
            # metric_counter 内求平均值
            tq.set_postfix(loss_acc=self.metric_counter.loss_message())
           
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch, validation=valid)

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
    

if __name__ == "__main__":
    with open('networks/config/config.yaml', 'r') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)
        data_dir = config_list['nonsky_dir']
        batch_size = config_list['batch_size']
        train_imgsize = config_list['image_size']

    train_dir = os.path.join(data_dir, "train")
    train_transform = get_transform(train_imgsize, 'Resize')
    custom_dataset = ImagePMSet(root=train_dir, transform=train_transform)
    # 先记录下来，后面再写入文件. 42.245478541401894, 15.92258928945192
    PM_mean, PM_std = custom_dataset.get_mean_std()

    train_size = int(len(custom_dataset) * 7 / 8)  # 7:1
    valid_size = len(custom_dataset) - train_size
    train_data, valid_data = random_split(custom_dataset, [train_size, valid_size])
    print("Train data length:{}, Val data length:{}".format(len(train_data), len(valid_data)))

    # 后续若 Subset 使用不同的 transform 如何处理？
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size)

    trainer = Trainer(config_list, train=train_loader, val=valid_loader)
    trainer.train()








