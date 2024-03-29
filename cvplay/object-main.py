#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

from multiprocessing.spawn import prepare
import pandas as pd
from matplotlib import pyplot as plt, patches

import torch
from torch.utils.data import random_split, DataLoader
from torch import optim
from torchsummary import summary

import numpy as np

from FRCNN import get_model
from bustruck import BusTruckDataset, prepare_dataset

# input params
IMAGE_ROOT = 'E:\\datasets\\bus-trucks\\images\\images'
#IMAGE_ROOT = '/Users/liangkun/dataset/bus-trucks/images'
DF_PATH = 'E:\\datasets\\bus-trucks\\df.csv'
#DF_PATH = '/Users/liangkun/dataset/bus-trucks/df.csv'
NAME2LABEL = {'BG':0, 'Truck':1, 'Bus':2}
LABEL2NAME = ['BG', 'Truck', 'Bus']
NUM_CLASSES = len(NAME2LABEL)
IMAGE_H = 224
IMAGE_W = 224
N_SAMPLES = -1

def collate_fn(batch):
    return tuple(zip(*batch))

def plot_samples(ds, nrow, ncol, figsize=(6, 6)):
    # plot a image to verify
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        image, targets = ds[i]
        labels = targets["labels"]
        bbs = targets["boxes"]
        ax.imshow(image.permute(1,2,0))
        for j in range(len(bbs)):
            bb = bbs[j]
            label = labels[j]
            xmin, ymin, xmax, ymax = bb[0], bb[1], bb[2], bb[3]
            w = xmax - xmin
            h = ymax - ymin
            ax.add_patch(patches.Rectangle((xmin, ymin), w, h, fill=False, edgecolor='green', lw=2))
            ax.text(xmin, ymin, LABEL2NAME[label], color='red', fontsize=10, weight='bold')
    plt.show()

def train_batch(xs, ys, model, optimizer):
    model.train()
    optimizer.zero_grad()
    losses = model(xs, ys)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def validate_batch(xs, ys, model):
    model.train()
    losses = model(xs, ys)
    loss = sum(loss for loss in losses.values())
    return loss.item()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds, val_ds, test_ds = prepare_dataset(DF_PATH, IMAGE_ROOT, IMAGE_W, IMAGE_H, N_SAMPLES, NAME2LABEL)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model = get_model(NUM_CLASSES)
    model = model.to(device)
    #summary(model, input_size=(3, 224, 224))

    optimizer = optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    ls_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)
    # trainning loop
    train_losses = []
    val_losses = []
    for epoch in range(20):
        start = time.time()
        train_epoch_losses = []
        for xs, ys in train_dl:
            nxs = [x.to(device) for x in xs]
            nys = [{k:v.to(device) for k, v in y.items()} for y in ys]
            loss = train_batch(nxs, nys, model, optimizer)
            train_epoch_losses.append(loss)
        train_loss = np.mean(train_epoch_losses)
        train_losses.append(train_loss)

        val_epoch_losses = []
        for xs, ys in val_dl:
            nxs = [x.to(device) for x in xs]
            nys = [{k:v.to(device) for k, v in y.items()} for y in ys]
            loss = validate_batch(nxs, nys, model)
            val_epoch_losses.append(loss)
        val_loss = np.mean(val_epoch_losses)
        ls_scheduler.step(val_loss)
        val_losses.append(val_loss)
        torch.save(model, 'frcnn.pth')
        consume = time.time() - start
        print("epcho: %d, consume: %ds, train loss: %.4f, val loss: %.4f" % (epoch, consume, train_loss, val_loss))
