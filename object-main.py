#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

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
    image_gtbbs.extend(zip(xmins, ymins, xmaxs, ymaxs))

