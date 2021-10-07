#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pickle

def load_cifar_batch(fname):
    with open(fname, 'rb') as inf:
        raw = pickle.load(inf, encoding='bytes')
        labels = F.one_hot(torch.tensor(raw[b'labels']), num_classes=10)
        imgs = torch.tensor(raw[b'data'] / 255).float().reshape((-1, 3, 32, 32))
        return imgs, labels

def load_label_names(fname):
    with open(fname, 'rb') as inf:
        label_names = pickle.load(inf, encoding='bytes')
        return [x.decode('utf-8') for x in label_names[b'label_names']]

class CifarDataset(Dataset):
    def __init__(self, basename):
        train_xs = []
        train_ys = []
        for name in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']:
            x, y = load_cifar_batch(basename + '/' + name)
            train_xs.append(x)
            train_ys.append(y)
        self.train_x = torch.cat(train_xs, axis=0)
        self.train_y = torch.cat(train_ys, axis=0)
        self.test_x, self.test_y = load_cifar_batch(basename + '/test_batch')
        self.label_names = load_label_names(basename + '/batches.meta')
    
    def __len__(self):
        return self.train_x.shape[0]
    
    def __getitem__(self, ix):
        return self.train_x[ix], self.train_y[ix]

