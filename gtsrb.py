#!/usr/bin/env python
#-*- coding:utf-8 -*-

from PIL import Image
import torch
import pandas as pd
import os
import glob
from torch.utils.data import Dataset

class GTSRB(Dataset):
    def __init__(self, dir, train=True, meta=None, transforms=None, testtransforms=None):
        super().__init__()
        self.train = train
        if train:
            self.files = glob.glob(os.path.join(dir, '*', '*.ppm'))
        else:
            self.files = glob.glob(os.path.join(dir, '*.ppm'))

        if meta:
            self.id2name, self.name2id = GTSRB.parse_meta(meta)
        self.transforms = transforms
        self.testtransforms = testtransforms
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, ix):
        path = self.files[ix]
        img = Image.open(path)
        # img = cv2.resize(img, (32, 32))
        # img = img / 255.0
        # img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)
        label = -1
        if self.train and self.id2name:
            parent = os.path.dirname(path)
            label_str = os.path.basename(parent)
            label = self.name2id[self.id2name[label_str]]
        return img, label
    
    def collate_fn(self, batch):
        imgs, labels = list(zip(*batch))
        if self.transforms:
            imgs = [self.transforms(img).unsqueeze(0) for img in imgs]
        imgs = torch.cat(imgs)
        labels = torch.tensor(labels)
        return imgs, labels
    
    def test_collate_fn(self, batch):
        imgs, labels = list(zip(*batch))
        if self.testtransforms:
            imgs = [self.testtransforms(img).unsqueeze(0) for img in imgs]
        imgs = torch.cat(imgs)
        labels = torch.tensor(labels)
        return imgs, labels

    @staticmethod
    def parse_meta(meta):
        classids = pd.read_csv(meta)
        classids.set_index('ClassId', inplace=True)
        classids = classids.to_dict()['SignName']
        id2name = {f'{k:05d}':v for k, v in classids.items()}
        name2id = {v:int(k) for k, v in classids.items()}
        return id2name, name2id