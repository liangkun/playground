#!/usr/bin/env
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torchvision import models

class FRCNN(nn.Module):
    r"""Faster R-CNN model implementation.
    """
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)