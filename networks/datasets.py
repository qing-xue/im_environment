import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import re
import glob


class ImagePMSet(data.Dataset):
    def __init__(self, root, transform):
        paths_mask = root + '/*/*.bmp'
        img_paths = glob.glob(paths_mask)

        self.imgs, self.PMs = [], []
        self.transforms = transform

        for k in img_paths:
            self.imgs.append(k)
            PM25 = int(re.split('[-.]', k)[-2])
            self.PMs.append(PM25)

        self.PM_mean = np.mean(self.PMs)
        self.PM_std  = np.std(self.PMs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        PM25 = self.PMs[index]
        pil_img = Image.open(img_path)

        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)

        # normalize the PM2.5 data
        data_y = (PM25 - self.PM_mean) / self.PM_std
        return data, data_y

    def __len__(self):
        return len(self.imgs)

    def inverse_PM(self, PMs):
        return PMs * self.PM_std + self.PM_mean


class ImgDataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.label_name = {"L0": 0, "L1": 1, "L2": 2}  # change 0 to L0
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self, data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # traverse class
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.bmp'), img_names))

                # traverse photos
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_img, int(label)))

        if len(data_info) == 0:
            raise Exception("\n data_dir:{} is a empty dir! Please checkout your path to images!".format(data_dir))
        return data_info
