#!/usr/bin/env
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from glob import glob
from random import shuffle

import cv2

class CatsDogs(Dataset):
    def __init__(self, folder, transform=None):
        cat_fnames = glob(folder + '/cats/*.jpg')
        dog_fnames = glob(folder +'/dogs/*.jpg')
        self.fnames = cat_fnames + dog_fnames
        shuffle(self.fnames)
        self.labels = [fname.split('/')[-1].startswith('dog') for fname in self.fnames]
        self.transform = transform
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, ix):
        fname = self.fnames[ix]
        label = self.labels[ix]
        img = cv2.imread(fname)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img/255, dtype=torch.float).permute(2, 0, 1)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float)
