"""
==========================
切分原始图像制作图像块数据集
==========================

需要在 Results/ 目录下包含所有的原始图像块；从 'crop_labels.xlsx' 中
切分出单条的标签信息存放至标签文件夹下。

"""
import glob
import os
import numpy as np
import re
import pandas as pd
from PIL import Image
from utils import ImageMapper, get_excel_data


class ImageCropper():
    
    def __init__(self, im):
        self.im = im

    def crop(self, box_w=256, box_h=256, stride_w=256, stride_h=256, epsilon=10):
        """Crop image to get patches.

        :param epsilon: 右下方边界容忍值，低于之则直接丢弃
        :return: 返回截取的 patches 以及其对应于原图的坐标
        """
        im = self.im
        width = im.size[0]
        height = im.size[1]
        if width < box_w or height < box_h:
            return

        patches, patches_idx = [], []
        iw = np.arange(0, width  - box_w + 1, stride_w)
        jh = np.arange(0, height - box_h + 1, stride_h)
        for i in iw:
            for j in jh:
                box = (i, j, i + box_w, j + box_h)
                cm = im.crop(box)
                patches.append(cm) 
                patches_idx.append(box)
        # repair x and y orientation's boundary
        if width - box_w - iw[-1] > epsilon:
            for j in jh:
                box = (width - box_w, j, width, j + box_h)
                cm = im.crop(box)
                patches.append(cm) 
                patches_idx.append(box)
        if height - box_h - jh[-1] > epsilon:
            for i in iw:
                box = (i, height - box_h, i + box_w, height)
                cm = im.crop(box)
                patches.append(cm) 
                patches_idx.append(box)
        # need only once
        if width - box_w - iw[-1] > epsilon and height - box_h - jh[-1] > epsilon:
            box = (width - box_w, height - box_h, width, height)
            cm = im.crop(box)
            patches.append(cm) 
            patches_idx.append(box)

        return patches, patches_idx


# 获取误差较大的图片名
def get_remove_file(filename):
    data = []
    f = open(filename, 'r', encoding="utf-8")
    file_data = f.readlines()
    zh2En = str.maketrans("'上午''下午'", "'AM''PM'", '\\')
    for i in file_data:
        i = i.translate(zh2En)
        i = i.replace("\n", "")
        data.append(i)
    return data


def judge_is_sky(img, threshold):
    grey_img = img.convert('L')
    grey_img_array = np.array(grey_img)
    shape = grey_img_array.shape
    mean = np.mean(grey_img_array)
    var = np.var(grey_img_array)
    # print(var)
    if(var < threshold):
        return True
    else:
        return False


def process_crop(Heshan_imgset = r'F:\workplace\public_dataset\Heshan_imgset'):
    types = ('*.jpg', '*.png') 
    img_paths = []
    for files in types:
        paths_mask = Heshan_imgset + '/*/*/' + files
        img_paths.extend(glob.glob(paths_mask))
 
    result_folder = os.path.join(Heshan_imgset, 'Results') 
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    data_label = pd.DataFrame()
    remove_image_list = get_remove_file(r'data/remove_image.txt')
    for i, filename in enumerate(img_paths):
        # 剔除误差较大的图片
        cur_im = os.path.split(filename)[1]
        if cur_im in remove_image_list:
            continue
        im = Image.open(filename)
        imCropper = ImageCropper(im)
        patches, boxes = imCropper.crop(box_w=256, box_h=256, stride_w=256, stride_h=256)
        
        # 拍摄时间--PM2.5值
        ch2En = str.maketrans("'上午''下午'", "'AM''PM'")
        img_id = re.split('[/\\\\.]', filename)[-2]
        shot_time = img_id.translate(ch2En)[4:]
        row_data = imageMapper.get_row(img_id)
        PM_25 = str(round(row_data['PM2.5'].values[0]))
        # 图块位置--天空/非天空
        threshold = 120     # 划分天空区域的方差阈值
        for i in range(len(patches)):
            im_patch = patches[i]
            box = boxes[i]
            ref_origin = "({},{})".format(box[0], box[1])
            if judge_is_sky(im_patch, threshold):
                is_sky = '0'
            else:
                is_sky = '1'

            patch_name = '-'.join([shot_time, ref_origin, is_sky, PM_25]) + '.bmp'
            print(os.path.join(result_folder, patch_name))
            im_patch = np.array(im_patch, dtype='uint8')
            im_patch = Image.fromarray(im_patch)
            im_patch.save(os.path.join(result_folder, patch_name), 'bmp')
            new_row = row_data
            new_row.loc[0, 'IMG_ID'] = patch_name
            data_label = data_label.append(new_row, ignore_index=True)

    writer = pd.ExcelWriter('crop_labels.xlsx')
    data_label.to_excel(writer, float_format='%.5f')
    writer.save()


if __name__ == '__main__':
    obj_data_path = r'..\客观图像质量指标测定-李展20210218.xlsx'
    sbj_data_path = r'..\志清雪清主观数据标定及加和分析20210218.xlsx'

    df_data = get_excel_data(obj_data_path, sbj_data_path)
    imageMapper = ImageMapper(df_data)

    process_crop()
