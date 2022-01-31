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

def get_model(num_classes):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
    return model