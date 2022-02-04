#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset

import os
import glob
import cv2
from random import randint

class SegDataset(Dataset):
    def __init__(self, root, split, tfms):
        self.imgpath = f"{root}/images_prepped_{split}"
        self.annpath = f"{root}/annotations_prepped_{split}"
        self.imgs = glob.glob(self.imgpath + '/*')
        self.tfms = tfms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        imgfile = self.imgs[ix]
        img = cv2.imread(imgfile)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgbasename = os.path.basename(imgfile)
        maskfile = f"{self.annpath}/{imgbasename}"
        mask = cv2.imread(maskfile)
        mask = cv2.resize(mask, (224, 224))
        
        return img, mask
    
    def choose(self):
        return self[randint(len(self))]
    
    def collate_fn(self, batch):
        imgs, masks = zip(*batch)
        xs = torch.cat([self.tfms(img).unsqueeze(0) for img in imgs])
        ys = torch.cat([torch.Tensor(mask).permute(2, 1, 0)[0].long().unsqueeze(0) for mask in masks])
        return xs, ys