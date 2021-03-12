"""
============================
天空/非天空图像块和标签分开存储
============================

需要在 Results/ 目录下包含所有的原始图像块；从 'crop_labels.xlsx' 中
切分出单条的标签信息存放至标签文件夹下。

"""
import os
import re
import shutil

import pandas as pd


def mkdir(floder):
    if not os.path.exists(floder):
        os.makedirs(floder)


# 将 Results/ 下的原始图像块归入 sky/ 和 non_sky/ 文件夹
def standardize_data_folder():
    sky_folder = os.path.join(patch_dir, 'sky') 
    nonsky_folder = os.path.join(patch_dir, 'non_sky') 
    mkdir(sky_folder)
    mkdir(nonsky_folder)

    for filename in os.listdir(patch_dir):
        if not filename.endswith('.bmp'):
            continue
        is_sky = re.split('[-]', filename)[-2]
        # 0 表示天空，1 表示其他
        if '0' == is_sky:
            shutil.move(os.path.join(patch_dir, filename), sky_folder)
            print(filename, " ==> ", sky_folder)
        else:
            shutil.move(os.path.join(patch_dir, filename), nonsky_folder)
            print(filename, " ==> ", nonsky_folder)
    

# 将 excel 里的标签切分为单个文本文件
def standardize_label_folder():
    skylabel_folder = os.path.join(patch_dir, 'sky_labels') 
    nonsky_labels_folder = os.path.join(patch_dir, 'non_sky_labels') 
    mkdir(skylabel_folder)
    mkdir(nonsky_labels_folder)

    pd_labels = pd.read_excel('crop_labels.xlsx', index_col=[0])
    for i in range(len(pd_labels)):
        row_data = pd_labels.iloc[i]
        img_id = row_data['IMG_ID']
        is_sky = re.split('[-]', img_id)[-2]
        # 0 表示天空，1 表示其他
        if '0' == is_sky:
            filename = '{}.csv'.format(os.path.join(skylabel_folder, img_id))
            row_data.to_csv(filename, index=True)
            print(img_id, " ==> ", skylabel_folder)
        else:
            filename = '{}.csv'.format(os.path.join(nonsky_labels_folder, img_id))
            row_data.to_csv(filename, index=True)
            print(img_id, " ==> ", nonsky_labels_folder)


patch_dir = r'F:\workplace\public_dataset\Heshan_imgset\Results'

if __name__ == '__main__':
    standardize_label_folder()