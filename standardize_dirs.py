import os
import shutil
import re

def mkdir(floder):
    if not os.path.exists(floder):
        os.makedirs(floder)

# 将 Results/ 下的原始图像块归入 sky/ 和 non_sky/ 文件夹
patch_dir = r'F:\workplace\public_dataset\Heshan_imgset\Results'
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
    