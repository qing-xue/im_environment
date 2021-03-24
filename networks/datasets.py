import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import re


class ImagePMSet(data.Dataset):
    def __init__(self, root, transform):
        # 所有图片的绝对路径
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)

        PM25 = int(re.split('[-.]', img_path)[-2])
        return data, PM25

    def __len__(self):
        return len(self.imgs)