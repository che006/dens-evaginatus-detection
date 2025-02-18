import torch
import torch.nn as nn
import torch.optim as optim

import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import os
import numpy as np
import torch.nn.functional as F

import random
import shutil
from torchvision.models import vgg16


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Load the pre-trained VGG16 model
        vgg = vgg16(pretrained=False)
        
        # Remove the last max-pooling layer
        self.features = nn.Sequential(*list(vgg.features.children())[:-1])

        # Add the fully connected layer for regression task
        # The input size (258048) is determined by the output shape of the features part
        self.classifier = nn.Linear(258048, 8)  # Output 4 sets of xy coordinates

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
class SimpleCNN_classification(nn.Module):
    def __init__(self):
        super(SimpleCNN_classification, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), #0
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),#2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), #5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), #7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), #10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), #12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), #14
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x