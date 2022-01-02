#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import torch

IMAGE_ROOT = 'F:\\datasets\\bus-trucks\\images\\images'
DF_PATH = 'F:\\datasets\\bus-trucks\\df.csv'
NAME2LABEL = {'BG':0, 'Truck':1, 'Bus':2}

# prepare argument for BusTruckDataset
raw = pd.read_csv(DF_PATH)
image_ids = raw['ImageID'].unique().tolist()
image_labels = []
image_gtbbs = []

for image_id in image_ids:
    imagedf = raw[raw['ImageID'] == image_id]
    image_labels.append([NAME2LABEL[name] for name in imagedf['LabelName']])
    xmins = imagedf['XMin'].tolist()
    ymins = imagedf['YMin'].tolist()
    xmaxs = imagedf['XMax'].tolist()
    ymaxs = imagedf['YMax'].tolist()
    image_gtbbs.append(zip(xmins, ymins, xmaxs, ymaxs))

image_labels = torch.tensor(image_labels, dtype=torch.int)
image_gtbbs = torch.tensor(image_gtbbs, dtype=torch.float32)

print(len(image_ids))
print(image_ids[0])
print(image_labels.shape)
print(image_labels[0])
print(image_gtbbs.shape)
print(image_gtbbs[0])
