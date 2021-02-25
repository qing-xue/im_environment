import glob
import random
import os
import numpy as np
import re
from PIL import Image, ImageDraw
from utils import ExcelMapper

excelMapper = ExcelMapper(r'..\客观图像质量指标测定-李展20210218.xlsx')

#=================== Utils ===================#
def im_crop(im, box_w=256, box_h=256, stride_w=256, stride_h=256, epsilon=10):
    """Crop image to get patches.

    :param epsilon: 右下方边界容忍值，低于之则直接丢弃
    :return: 返回截取的 patches 以及其对应于原图的坐标
    """
    width = im.size[0]
    height = im.size[1]
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

#=================== Tests ===================#
def test_im_crop():
    im = Image.open('20190927am1.jpg')
    patches, boxes = im_crop(im, box_w=512, box_h=512, stride_w=512, stride_h=512)
    draw = ImageDraw.Draw(im)
    for box in boxes:
        color = (random.randint(64,255), random.randint(64,255), random.randint(64,255))
        draw.rectangle(box, outline=color, width=5)
        # im.show()
    im.show()

#=================== Process ===================#
def process_crop():
    Heshan_imgset = 'F:/workplace/public_dataset/Heshan_imgset'
    paths_mask = Heshan_imgset + '/*/*/*[jpg, png, jpeg]'
    img_paths = glob.glob(paths_mask)
    
    # Results
    result_folder = Heshan_imgset + '/Results'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Process
    for i, filename in enumerate(img_paths):
        im = Image.open(filename)
        patches, boxes = im_crop(im, box_w=512, box_h=512, stride_w=512, stride_h=512)

        # 拍摄时间--PM2.5值
        ch2En = str.maketrans("'上午''下午'", "'AM''PM'")
        img_id = re.split('[/\\\\.]', filename)[-2]
        shot_time = img_id.translate(ch2En)[4:]
        PM_25 = str(excelMapper.get_PM2_5(img_id))

        # 图块位置--天空/非天空
        for i in range(len(patches)):
            im_patch = patches[i]
            box = boxes[i]
            ref_origin = str(box[0]) + '_' + str(box[1])
            is_sky = '1'    # bug...

            # save
            patch_name = shot_time + '-' + ref_origin + '-' + is_sky + '-' + PM_25 + '.bmp'  # ugly
            print(patch_name)
            im_patch = np.array(im_patch, dtype='uint8')
            im_patch = Image.fromarray(im_patch)
            im_patch.save(os.path.join(result_folder, patch_name), 'bmp')

if __name__ == '__main__':
    # test_im_crop()    # 查看图片切块效果
    process_crop()