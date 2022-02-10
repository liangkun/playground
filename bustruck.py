#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, random_split
import os
import cv2
import pandas as pd

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
    def __init__(self, image_root, image_ids, labels, gtbbs, h=224, w=224):
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
        targets = {
            "boxes":self.gtbbs[ix],
            "labels":self.labels[ix]
        }

        return image, targets
    
    def collate_fn(self, batch):
        pass

def prepare_dataset(df_path, image_root, image_w, image_h, nsamples, name2label):
    # Prepare BusTruckDataset
    raw = pd.read_csv(df_path)
    image_ids = raw['ImageID'].unique().tolist()[:nsamples]
    image_labels = []
    image_gtbbs = []

    for image_id in image_ids:
        imagedf = raw[raw['ImageID'] == image_id]
        image_labels.append(
            torch.tensor([name2label[name] for name in imagedf['LabelName'].values.tolist()])
        )
        boxes = imagedf[['XMin', 'YMin', 'XMax', 'YMax']].values
        boxes[:, [0, 2]] *= image_w
        boxes[:, [1, 3]] *= image_h
        image_gtbbs.append(torch.Tensor(boxes).float())

    samples = BusTruckDataset(image_root, image_ids, image_labels, image_gtbbs, h=image_h, w=image_w)
    n_samples = len(samples)
    n_train = int(n_samples * 0.7)
    n_val = (n_samples - n_train) // 2
    n_test = n_samples - n_train - n_val

    train_ds, val_ds, test_ds = random_split(samples, [n_train, n_val, n_test])
    print('trainset: %d, valset: %d, testset: %d' % (len(train_ds), len(val_ds), len(test_ds)))
    return train_ds, val_ds, test_ds