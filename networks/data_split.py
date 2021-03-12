"""划分训练、验证集

并按照 datasets.ImageFolder 的方式组织目录
"""

import os
import utils
import re
import shutil
import yaml


def add_to_path(dst_path, pollution):
    """在目标路径之后追加标签目录 """
    labels = list(pollution.keys())
    if PM25 <= pollution[labels[0]]: 
        dst_path = os.path.join(dst_path, labels[0])
    elif PM25 <= pollution[labels[1]]:
        dst_path = os.path.join(dst_path, labels[1])
    elif PM25 <= pollution[labels[2]]:
        dst_path = os.path.join(dst_path, labels[2])
    return dst_path


# 仅使用取整后的 PM2.5 值
with open(r'.\config\config.yaml') as file:
    config_list = yaml.load(file, Loader=yaml.FullLoader)
    datax_dir = config_list['nonsky_dir']

train_folder = os.path.join(datax_dir, 'train') 
val_folder = os.path.join(datax_dir, 'val') 
utils.mkdir(train_folder)
utils.mkdir(val_folder)

pollution = {
    'L0': 35,
    'L1': 70,
    'L2': 100
}
for y in pollution.keys():
    utils.mkdir(os.path.join(train_folder, y))
    utils.mkdir(os.path.join(val_folder, y))

# 切分：11月6号前的作为训练集，往后的作为验证集
split_date = int(1106)
for filename in os.listdir(datax_dir):
    if not filename.endswith('.bmp'):
        continue
    date = int(filename[:4])
    PM25 = int(re.split('[-.]', filename)[-2])
    if date < split_date:
        dst_path = add_to_path(train_folder, pollution)
        shutil.move(os.path.join(datax_dir, filename), dst_path)
        print(os.path.join(datax_dir, filename), "==>", dst_path)
    else:
        dst_path = add_to_path(val_folder, pollution)
        shutil.move(os.path.join(datax_dir, filename), dst_path)
        print(os.path.join(datax_dir, filename), "==>", dst_path)




