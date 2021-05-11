import logging
from collections import defaultdict

import numpy as np
from tensorboardX import SummaryWriter

WINDOW_SIZE = 100


class MetricCounter:
    def __init__(self, exp_name):
        self.writer = SummaryWriter(exp_name)
        logging.basicConfig(filename='{}.log'.format(exp_name), level=logging.DEBUG)
        self.metrics = defaultdict(list)
        self.images = defaultdict(list)
        self.best_metric = 0

    def add_image(self, x: np.ndarray, tag: str):
        self.images[tag].append(x)

    def clear(self):
        self.metrics = defaultdict(list)
        self.images = defaultdict(list)

    def add_losses(self, l_mse):
        for name, value in zip(('MSE_loss',),
                               (l_mse,)):
            self.metrics[name].append(value)

    def add_metrics(self, acc):
        for name, value in zip(('Acc',),
                               (acc,)):
            self.metrics[name].append(value)

    def loss_message(self):
        # ('MSE_loss',) 可设置为私有变量，为何作者不输出 Acc？
        metrics = ((k, np.mean(self.metrics[k][-WINDOW_SIZE:])) for k in ('MSE_loss', 'Acc'))
        return '; '.join(map(lambda x: f'{x[0]}={x[1]:.4f}', metrics))

    def write_to_tensorboard(self, epoch_num, validation=False):
        scalar_prefix = 'Validation' if validation else 'Train'
        for tag in ('MSE_loss', 'Acc'):
            self.writer.add_scalar(f'{scalar_prefix}_{tag}', np.mean(self.metrics[tag]), global_step=epoch_num)
        for tag in self.images:
            imgs = self.images[tag]
            if imgs:
                imgs = np.array(imgs)
                self.writer.add_images(tag, imgs[:, :, :, ::-1].astype('float32') / 255, dataformats='NHWC',
                                       global_step=epoch_num)
                self.images[tag] = []

    def update_best_model(self):
        cur_metric = np.mean(self.metrics['Acc'])
        if self.best_metric < cur_metric:
            self.best_metric = cur_metric
            return True
        return False
