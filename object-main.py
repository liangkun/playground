#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import torch

IMAGE_ROOT = 'F:\\datasets\\bus-trucks\\images\\images'
#DF_PATH = 'F:\\datasets\\bus-trucks\\df.csv'
DF_PATH = './df.csv'
NAME2LABEL = {'BG':0, 'Truck':1, 'Bus':2}
IMAGE_H = 224
IMAGE_W = 224
N_SAMPLES = 50

# prepare argument for BusTruckDataset
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

print(len(raw))
print(len(image_ids))
print(image_ids[0])
print(len(image_labels))
print(image_labels[0])
print(len(image_gtbbs))
print(image_gtbbs[0])
