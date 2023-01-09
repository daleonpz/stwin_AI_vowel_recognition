# create a pytorch model with two conv2d - relu - maxpool layers

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fc import FC

import logging

logger = logging.getLogger(__name__)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1) # 3 input channels, 24 output channels, 3x3 kernel
        # equation for output conv layer size: (W-F+2P)/S + 1 # W = input size, F = filter size, P = padding, S = stride
        # (20-3+2)/1 + 1 = 20
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24,48, kernel_size=3, stride=1, padding=1) # 24 input channels, 48 output channels, 3x3 kernel
        self.bn2 = nn.BatchNorm2d(48)
        self.pool = nn.MaxPool2d(2, 2)  # Downsample the input image by a factor of 2
        # equation for output maxpool layer size: (W-F)/S + 1 # W = input size, F = filter size, S = stride
        # (20-2)/2 + 1 = 10

        self.fc = FC(10*10*48, 2) # 10x10x48 input features, 2 output features (for 2 classes)
        logger.debug("CNN model created")
        logger.debug("CNN model: %s" % self)

    def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn1(self.conv1(x)))
        logger.debug("CNN model: conv1 output shape: %s" % str(x.shape))
        x = F.relu(self.bn2(self.conv2(x)))
        logger.debug("CNN model: conv2 output shape: %s" % str(x.shape))
        x = self.pool(x)
        logger.debug("CNN model: pool output shape: %s" % str(x.shape))
        x = self.fc(x)

        return x



