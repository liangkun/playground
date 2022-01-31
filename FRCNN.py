#!/usr/bin/env
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FRCNN(nn.Module):
    r"""Faster R-CNN model implementation.
    """
    def __init__(self):
        super().__init__()
        vgg11 = models.vgg11_bn(pretrained=True)
        for param in vgg11.features.parameters():
            param.requires_grad = False
        
        self.backbone = vgg11.features[:-1]
    
    def forward(self, xs):
        res = self.backbone(xs)
        return res

def get_model(name='frcnn'):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    return model