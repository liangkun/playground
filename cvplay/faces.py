#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import glob
import os
import pandas as pd
import cv2
from copy import deepcopy
import numpy as np

class FaceData(Dataset):
    def __init__(self, rootdir, test=False):
        super().__init__()
        self.rootdir = rootdir
        if test:
            datadir = os.path.join(rootdir, 'data', 'test')
            metafile = os.path.join(rootdir, 'data', 'test_frames_keypoints.csv')
        else:
            datadir = os.path.join(rootdir, 'data', 'training')
            metafile = os.path.join(rootdir, 'data', 'training_frames_keypoints.csv')
        self.datadir = datadir
        self.meta = pd.read_csv(metafile)
    
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        imgpath = os.path.join(self.datadir, self.meta.iloc[index, 0])
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kp = deepcopy(self.meta.iloc[index, 1:].tolist())
        kp_x = (np.array(kp[0::2])/img.shape[1]).tolist()
        kp_y = (np.array(kp[1::2])/img.shape[0]).tolist()
        kp2 = torch.tensor(kp_x + kp_y, dtype=torch.float)
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img/255, dtype=torch.float)
        img = img.permute(2, 0, 1)
        return img, kp2
    