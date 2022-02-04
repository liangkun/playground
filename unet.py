#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision.models import vgg16_bn

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )

class Unet(nn.Module):
    def __init__(self, pretrained=True, out_channels=12):
        super().__init__()
        self.encoder = vgg16_bn(pretrained=True).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])
        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = conv(512, 1024)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = conv(64 + 128, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, xs):
        block1 = self.block1(xs)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        bottleneck = self.bottleneck(block5)
        features = self.conv_bottleneck(bottleneck)

        features = self.up_conv6(features)
        features = torch.cat([features, block5], dim=1)
        features = self.conv6(features)

        features = self.up_conv7(features)
        features = torch.cat([features, block4], dim=1)
        features = self.conv7(features)

        features = self.up_conv8(features)
        features = torch.cat([features, block3], dim=1)
        features = self.conv8(features)

        features = self.up_conv9(features)
        features = torch.cat([features, block2], dim=1)
        features = self.conv9(features)

        features = self.up_conv10(features)
        features = torch.cat([features, block1], dim=1)
        features = self.conv10(features)

        features = self.conv11(features)
        return features
    