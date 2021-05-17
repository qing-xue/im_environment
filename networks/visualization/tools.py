import os
from PIL import Image
from matplotlib import pyplot as plt


if __name__ == '__main__':
    img_list = ['c_0_iter_299_loss_-27.81174087524414.jpg',
                'c_1_iter_299_loss_-14.940608978271484.jpg',
                'c_2_iter_299_loss_-15.837347030639648.jpg']
    for i in range(0, 3):
        path = r'.\generated\class_{}_blurfreq_4_blurrad_1_wd0.0001'.format(i)
        src = Image.open(os.path.join(path, img_list[i]))
        gray = src.convert('L')
        plt.subplot(2, 3, i+1)
        plt.imshow(src)
        plt.subplot(2, 3, i+1+3)
        plt.imshow(gray)

    plt.show()