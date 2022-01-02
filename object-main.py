#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import random_split

from bustruck import BusTruckDataset

# input params
IMAGE_ROOT = 'F:\\datasets\\bus-trucks\\images\\images'
DF_PATH = 'F:\\datasets\\bus-trucks\\df.csv'
#DF_PATH = './df.csv'
NAME2LABEL = {'BG':0, 'Truck':1, 'Bus':2}
IMAGE_H = 224
IMAGE_W = 224
N_SAMPLES = 50

# Prepare BusTruckDataset
raw = pd.read_csv(DF_PATH)
image_ids = raw['ImageID'].unique().tolist()[:N_SAMPLES]
image_labels = []
image_gtbbs = []

for image_id in image_ids:
    imagedf = raw[raw['ImageID'] == image_id]
    image_labels.append(
        [NAME2LABEL[name] for name in imagedf['LabelName'].values.tolist()]
    )
    boxes = imagedf[['XMin', 'YMin', 'XMax', 'YMax']].values
    boxes[:, [0, 2]] *= IMAGE_W
    boxes[:, [1, 3]] *= IMAGE_H
    image_gtbbs.append(torch.Tensor(boxes).float())

samples = BusTruckDataset(IMAGE_ROOT, image_ids, image_labels, image_gtbbs, h=IMAGE_H, w=IMAGE_W)
n_samples = len(samples)
n_train = int(n_samples * 0.7)
n_val = (n_samples - n_train) / 2
n_test = n_samples - n_train - n_val

train_ds, val_ds, test_ds = random_split(samples, [n_train, n_val, n_test])
print('trainset: %d, valset: %d, testset: %d' % (len(train_ds), len(val_ds), len(test_ds)))
