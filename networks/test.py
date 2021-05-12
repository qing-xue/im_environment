from __future__ import print_function, division
import time
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import yaml
from PIL import Image

from utils import value2class, im_segment, inverse_PM
from networks import vgg16_customize, get_nets
from datasets import ImagePMSet, get_transform

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Tester:
    def __init__(self, config, img: Image):
        self.config = config
        self.img = img

    def test(self):
        self._init_params()
        self.model.to(device)

        input_ = self.img
        transform = get_transform(self.config['image_size'], 'RandomCrop')

        time_start = time.time()
        self.model.to(device)
        self.model.train(False)
        self.model.eval()

        for i in range(0, 10):
            X = transform(input_)
            with torch.no_grad():
                X = X.unsqueeze(0)
                X = X.to(device)
                output = self.model(X)  # 可处理多个
                # outputs = map(inverse_PM, outputs)
                # outputs = map(value2class, outputs)

            _, preds = torch.max(output.data, 1)
            print(preds)

        time_end = time.time()
        print('Single image time cost {:.2f} s'.format(time_end - time_start))

    def _init_params(self):
        self.criterion = nn.MSELoss()  # get_loss 抽象
        self.model = get_nets(self.config['model'])
        # self.model.load_state_dict(torch.load(self.config['test']['model_path'], map_location=device))
        
        
if __name__ == '__main__':
    with open(r'networks\config\config.yaml') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)

    img_path = r'F:\workplace\public_dataset\Heshan_imgset\morning\1\20191116上午1.jpg'
    input_ = Image.open(img_path)
    tester = Tester(config_list, input_)
    tester.test()