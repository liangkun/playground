#!/usr/bine/env python
# -*- encoding: utf-8 -*-

from random import randint
import torch
from torch.utils.data import Dataset
from PIL import Image
from glob import glob

class AppleOrangeDataset(Dataset):
    def __init__(self, apple_dir, orange_dir, transforms=None, device="cuda"):
        super().__init__()
        self.apples = glob("%s/*" % apple_dir)
        self.oranges = glob("%s/*" % orange_dir)
        self.transforms = transforms
        if self.transforms is None:
            self.transforms = lambda x: x
    
        self.device = device

    def __len__(self):
        return len(self.apples)

    def __getitem__(self, ix):
        apple = self.apples[ix]
        orange = self.oranges[randint(0, len(self.oranges)-1)]

        apple = Image.open(apple).convert("RGB")
        orange = Image.open(orange).convert("RGB")

        return apple, orange
    
    def collate_fn(self, batch):
        srcs, trgs = list(zip(*batch))
        srcs = torch.cat([self.transforms(img)[None] for img in srcs]).to(self.device).float()
        trgs = torch.cat([self.transforms(img)[None] for img in trgs]).to(self.device).float()
        return srcs, trgs

