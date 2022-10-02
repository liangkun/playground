#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset

import os
import glob
import cv2
from random import randint

class SegDataset(Dataset):
    def __init__(self, root, split):
        self.imgpath = f"{root}/images_prepped_{split}"
        self.annpath = f"{root}/annotations_prepped_{split}"
        self.imgs = glob.glob(self.imgpath + '/*')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        imgfile = self.imgs[ix]
        img = cv2.imread(imgfile)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (torch.Tensor(img).float() / 255.0).permute(2, 0, 1)

        imgbasename = os.path.basename(imgfile)
        maskfile = f"{self.annpath}/{imgbasename}"
        mask = cv2.imread(maskfile)
        mask = cv2.resize(mask, (224, 224))
        mask = torch.Tensor(mask).long().permute(2, 0, 1)[0]
        
        return img, mask
    
    def choose(self):
        return self[randint(0, len(self))]
