import logging
from collections import defaultdict

import numpy as np
from tensorboardX import SummaryWriter

WINDOW_SIZE = 100  # 只计算最近的对应 batch 数目的指标的平均值


class MetricCounter:
    def __init__(self, exp_name):
        self.writer = SummaryWriter(exp_name)
        # 路径问题 data/
        logging.basicConfig(filename='{}.log'.format(exp_name), level=logging.DEBUG)
        self.metrics = defaultdict(list)
        self.images = defaultdict(list)
        self.best_metric = 0

    def add_image(self, x: np.ndarray, tag: str):
        self.images[tag].append(x)

    def clear(self):
        self.metrics = defaultdict(list)
        self.images = defaultdict(list)

    def add_losses(self, keys, values):
        # 自定义 ('MSE_loss',) 还是由外部传入
        for name, value in zip(keys, values):
            self.metrics[name].append(value)

    def add_metrics(self, keys, values):
        for name, value in zip(keys, values):
            self.metrics[name].append(value)

    def loss_message(self):
        metrics = ((k, np.mean(self.metrics[k][-WINDOW_SIZE:])) for k in self.metrics.keys())
        return '; '.join(map(lambda x: f'{x[0]}={x[1]:.4f}', metrics))

    def write_to_tensorboard(self, epoch_num, validation=False):
        scalar_prefix = 'Validation' if validation else 'Train'
        for tag in self.metrics.keys():
            # 比 self.metrics[k][-WINDOW_SIZE:]) 更精确的计算
            self.writer.add_scalars(tag, {scalar_prefix: np.mean(self.metrics[tag])}, global_step=epoch_num)
        for tag in self.images:
            imgs = self.images[tag]
            if imgs:
                imgs = np.array(imgs)
                self.writer.add_images(tag, imgs[:, :, :, ::-1].astype('float32') / 255, dataformats='NHWC',
                                       global_step=epoch_num)
                self.images[tag] = []
        self.writer.close()  # tensorboard --logdir=networks/classify/exp_resnet34_class/

    def update_best_model(self, key):
        cur_metric = np.mean(self.metrics[key])  # key = 'Acc'
        if self.best_metric < cur_metric:
            self.best_metric = cur_metric
            return True
        return False
