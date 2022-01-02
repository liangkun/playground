#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import os
import cv2

class BusTruckDataset(Dataset):
    r""" A Dataset for object detection.
    Arguments:
        image_root: directory where to find all images.
            All images must can be found at "${image_root}/${image_id}.jpg".
        image_ids: image ids
        labels: For each image_id, there is a array of lables corresponding to each gtbbs
        gtbbs: For each image_id, there is a array of gtbbs corresponding to each target.
            2D tensor of shape (K, 4). Each row contains [x_min, y_min, x_max, y_max] as ratio.
    """
    def __init__(self, image_root, image_ids, labels, gtbbs, h=224, w=224)):
        super.__init__()
        self.image_root = image_root
        self.image_ids = image_ids
        self.labels = labels
        self.gtbbs = gtbbs
        self.h = h
        self.w = w
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, ix):
        image_id = self.image_ids[ix]
        image = cv2.imread(os.path.join(self.image_root, image_id + '.jpg'))
        image = cv2.resize(image, (self.h, self.w))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image/255, dtype=torch.float32).permute(2, 0, 1)

        # convert gtbbs to absolute values
        gtbbs = self.gtbbs[ix]
        gtbbs[:, [0, 2]] *= self.w
        gtbbs[:, [1, 3]] *=self.h

        return image, self.labels[ix], gtbbs
    
    def collate_fn(self, batch):
        pass