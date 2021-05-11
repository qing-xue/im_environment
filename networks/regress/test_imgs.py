from __future__ import print_function, division
import time
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import yaml
from PIL import Image

import sys
from pathlib import Path
current_folder = Path(__file__).absolute().parent  # ugly
father_folder = str(current_folder.parent)
sys.path.insert(0, father_folder)

from utils import value2class, im_segment, inverse_PM
from models import vgg16_customize
from datasets import ImagePMSet, get_transform


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open(r'networks\config\config.yaml') as file:
    config_list = yaml.load(file, Loader=yaml.FullLoader)
    data_dir = config_list['nonsky_dir']
    val_fig = config_list['val']
    val_model = val_fig['model']
    val_imgsize = val_fig['imgsize']
    val_batch = val_fig['batch']


def test_single_img(model, input_):
    input_ = im_segment(input_)
    transform = get_transform(val_imgsize, 'RandomCrop')

    time_start = time.time()
    model.to(device)
    model.train(False)
    model.eval()

    for i in range(0, 10):
        X = transform(input_)
        with torch.no_grad():
            X = X.unsqueeze(0)
            X = X.to(device)
            output = model(X)  # 可处理多个
            # outputs = map(inverse_PM, outputs)
            # outputs = map(value2class, outputs)

        _, preds = torch.max(output.data, 1)
        # print(preds)

    time_end = time.time()
    print('Single image time cost {:.2f} s'.format(time_end - time_start))


# Load the model
vgg16 = models.vgg16_bn()
vgg16.load_state_dict(torch.load(val_model, map_location=device))

criterion = nn.MSELoss()
# eval_model_re(vgg16, criterion, valid_loader, valid_loader)
img_path = r'F:\workplace\public_dataset\Heshan_imgset\morning\1\20191116上午1.jpg'
input_ = Image.open(img_path)
test_single_img(vgg16, input_)