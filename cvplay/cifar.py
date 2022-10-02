#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pickle

def load_cifar_batch(fname):
    with open(fname, 'rb') as inf:
        raw = pickle.load(inf, encoding='bytes')
        labels = np.array(raw[b'labels'], dtype=np.long)
        imgs = raw[b'data'].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
        return imgs, labels

def load_label_names(fname):
    with open(fname, 'rb') as inf:
        label_names = pickle.load(inf, encoding='bytes')
        return [x.decode('utf-8') for x in label_names[b'label_names']]

class CifarDataset(Dataset):
    def __init__(self, basename, device=None, aug=None):
        self.device = device
        self.aug = aug
        train_xs = []
        train_ys = []
        for name in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']:
            x, y = load_cifar_batch(basename + '/' + name)
            train_xs.append(x)
            train_ys.append(y)
        self.train_x = np.concatenate(train_xs, axis=0)
        self.train_y = np.concatenate(train_ys, axis=0)
        self.label_names = load_label_names(basename + '/batches.meta')
    
    def __len__(self):
        return self.train_x.shape[0]
    
    def __getitem__(self, ix):
        return self.train_x[ix], self.train_y[ix]
    
    def collate_fn(self, batch):
        x, y = list(zip(*batch))
        if self.aug:
            x = self.aug.augment_images(x)
        x = torch.tensor(x)
        y = torch.tensor(y, dtype=torch.long)
        if self.device:
            x = x.to(self.device)
            y = y.to(self.device)
        x = (x / 255).permute(0, 3, 1, 2)  # normalize, change to pytorch
        return x, y

class CifarTestDataset(Dataset):
    def __init__(self, basename, device=None):
        self.device = device
        self.test_x, self.test_y = load_cifar_batch(basename + '/test_batch')
        self.test_x = torch.tensor(self.test_x)
        self.test_y = torch.tensor(self.test_y, dtype=torch.long)
        if self.device:
            self.test_x = self.test_x.to(self.device)
            self.test_y = self.test_y.to(self.device)
        self.test_x = (self.test_x/255).permute(0, 3, 1, 2)
    
    def __len__(self):
        return self.test_x.shape[0]
    
    def __getitem__(self, ix):
        return self.test_x[ix], self.test_y[ix]
