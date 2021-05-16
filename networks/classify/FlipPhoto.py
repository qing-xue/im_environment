"""图像增广
需先分好 0/1/2 三类文件夹
将 0 类变为原来的 2 倍， 2 类变为原来的 6 倍
"""
import cv2
import os
import sys


def rotate_img(degree, img_name, path_img):
    """
    degree: 角度(int)
    img_name: 图像名称
    path_img: 旋转图(.bmp)
    path_new_img: 输出路径
    """
    degree_str = str(degree)
    img = cv2.imread(path_img)
    new_img_name = img_name.replace(".bmp", "") + "-rotate_" + degree_str + ".bmp"

    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]

    # 定义一个旋转矩阵
    matRotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), degree, 1)  # mat rotate 1 center 2 angle 3 缩放系数
    new_img = cv2.warpAffine(img, matRotate, (height, width))

    path_new_img = path_img.replace(img_name, "") + new_img_name
    print(path_new_img)
    cv2.imwrite(path_new_img, new_img)


def main_func(data_dir):
    for root, dirs, _ in os.walk(data_dir):
        # 遍历类别
        for sub_dir in dirs:
            img_names = os.listdir(os.path.join(root, sub_dir))
            # img_names = list(filter(lambda x: x.endswith('.bmp'), img_names))  # 这一步耗时？

            # 遍历图片
            for i in range(len(img_names)):
                img_name = img_names[i]
                filter_list = ['rotate', 'flip']  # 避免重复增强图块
                if not img_name.endswith('.bmp') or any(key in img_name for key in filter_list):
                    continue

                path_img = os.path.join(root, sub_dir, img_name)
                grade = sub_dir

                # 分类 0 只需要翻转
                if 'L0' == grade:
                    img = cv2.imread(path_img)
                    horizontal_img = cv2.flip(img, 1)
                    new_img_name = img_name.replace(".bmp", "") + "-flip" + ".bmp"
                    path_new_img = os.path.join(root, sub_dir, new_img_name)
                    print(path_new_img)
                    cv2.imwrite(path_new_img, horizontal_img)

                # 分类 2 翻转后 + rotate1.5 + rotate-1.5
                elif 'L2' == grade:
                    img = cv2.imread(path_img)
                    horizontal_img = cv2.flip(img, 1)
                    new_img_name = img_name.replace(".bmp", "") + "-flip" + ".bmp"
                    path_new_img = os.path.join(root, sub_dir, new_img_name)
                    print(path_new_img)
                    cv2.imwrite(path_new_img, horizontal_img)

                    rotate_img(-1.5, img_name, path_img)
                    rotate_img(1.5, img_name, path_img)
                    rotate_img(-1.5, new_img_name, path_new_img)
                    rotate_img(1.5, new_img_name, path_new_img)


if __name__ == '__main__':
    # 'D:\BaiduNetdiskDownload\Proenviroment\Heshanimgset\Heshanimgset\Results\split1'
    img_folder = sys.argv[1]
    print('该目录下需包含 3类 子文件（L0/L1/L2），可以嵌套：', img_folder)  # 命令行传入文件路径
    main_func(img_folder)
