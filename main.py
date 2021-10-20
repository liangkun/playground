#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import cv2
import imgaug.augmenters as iaa
from torchsummary import summary

from cifar import CifarDataset, CifarTestDataset

def train_batch(x, y, model, lossfn, opt):
    model.train()
    yp = model(x)
    loss = lossfn(yp, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()

@torch.no_grad()
def val_loss(x, y, model, lossfn):
    model.eval()
    yp = model(x)
    loss = lossfn(yp, y)
    return loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    yp = model(x).argmax(dim=1)
    is_correct = (y == yp).cpu().numpy().tolist()
    return is_correct


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    dataset = CifarDataset('./data/cifar-10-batches-py', device=device)
    raw_test_dataset = CifarTestDataset('./data/cifar-10-batches-py', device=device)
    test_dataset, val_dataset = torch.utils.data.random_split(raw_test_dataset, [5000, 5000])
    print("training samples: %d" % len(dataset))
    print("validation samples: %d" % len(val_dataset))
    print("test samples: %d" % len(test_dataset))

    train_loader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(512*2*2, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    ).to(device)
    lossfn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True,
        factor=0.2, patience=5, threshold=1e-3, threshold_mode='rel', min_lr=5e-6)

    summary(model, torch.zeros(1, 3, 32, 32))

    train_losses = []
    val_losses = []
    val_acces = []
    for epoch in range(50):
        start = time.time()

        # training
        epoch_train_losses = []
        for x, y in train_loader:
            loss = train_batch(x, y, model, lossfn, optimizer)
            epoch_train_losses.append(loss)
        epoch_train_loss = np.mean(epoch_train_losses)
        train_losses.append(epoch_train_loss)

        epoch_val_losses = []
        epoch_val_acces = []
        for x, y in val_loader:
            loss = val_loss(x, y, model, lossfn)
            epoch_val_losses.append(loss)
            is_correct = accuracy(x, y, model)
            epoch_val_acces.extend(is_correct)
        epoch_val_loss = np.mean(epoch_val_losses)
        val_losses.append(epoch_val_loss)
        epoch_val_acc = np.mean(epoch_val_acces)
        val_acces.append(epoch_val_acc)
        scheduler.step(epoch_val_loss)

        end = time.time()
        print("epoch %d, %.2fs, trloss: %.2f, valloss: %.2f, valcc: %.2f" %
            (epoch, end-start, epoch_train_loss, epoch_val_loss, epoch_val_acc*100))

    test_acces = []
    for x, y in test_loader:
        is_correct = accuracy(x, y, model)
        test_acces.extend(is_correct)
    test_acc = np.mean(test_acces)
    print("======================")
    print("final test acc %.2f" % (test_acc*100))
    print("======================")

    plt.subplot(2, 1, 1)
    plt.plot(train_losses, 'b', label='train loss')
    plt.plot(val_losses, 'g', label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(val_acces, 'g', label='val acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

