import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import re


class ImagePMSet(data.Dataset):
    def __init__(self, root, transform):
        imgs = os.listdir(root)
        self.imgs, self.PMs = [], []
        self.transforms = transform

        for k in imgs:
            self.imgs.append(os.path.join(root, k))
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