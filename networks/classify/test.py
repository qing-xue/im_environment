import time
import torch
import torch.nn as nn
import yaml
from PIL import Image
import numpy as np
from scipy import stats
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
print(sys.path)  # 必要时检查，有时要进入脚本所在目录运行

from networks import get_nets
from datasets import get_transform

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Tester:
    def __init__(self, config, dataLoader: DataLoader):
        self.config = config
        # self.img = img  # config, img: Image
        self.dataLoader = dataLoader

    def test(self):
        self._init_params()
        self.model.to(device)
        self.model.train(False)
        self._test_single_img(self.img)

    def _test_single_img(self, input_):
        time_start = time.time()
        Xs = self.transform(input_)
        Xs = Xs.unsqueeze(0)
        for i in range(1, self.crop_img_blocks):
            X = self.transform(input_)
            Xs = torch.cat((Xs, X.unsqueeze(0)), 0)

        with torch.no_grad():
            Xs = Xs.to(device)
            outputs = self.model(Xs)  # 可处理多个
            _, preds = torch.max(outputs.data, 1)

        class_list = preds.cpu().data.numpy()
        print('Median class: {}'.format(np.median(class_list)))
        class_mode = stats.mode(class_list)
        print('Mode class: {}, count: {}/{}'.format(class_mode[0][0], class_mode[1][0], self.crop_img_blocks))

        time_end = time.time()
        print('Single image time cost {:.2f} s'.format(time_end - time_start))

    def _init_params(self):
        self.criterion = nn.MSELoss()  # get_loss 抽象
        self.model = get_nets(self.config['model']['g_name'], self.config['model']['out_features'])
        self.model.load_state_dict(torch.load(self.config['test']['model_path'], map_location=device)['model'])
        self.transform = transforms.RandomCrop(self.config['image_size'])
        self.crop_img_blocks = self.config['test']['crop_img_blocks']
        
        
if __name__ == '__main__':
    with open(r'..\config\config.yaml') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)
        data_dir = config_list['nonsky_dir']
        imgsize = config_list['image_size']

    img_path = r'F:\workplace\public_dataset\Heshan_imgset\morning\1\20191116上午1.jpg'
    img_path = r'D:\workplace\dataset\Heshan_imgset\Heshan_imgset\am_pm_123\filtering\test\L0\14.2-20191007上午2.jpg'
    input_ = Image.open(img_path)
    tester = Tester(config_list, input_)
    tester.test()

