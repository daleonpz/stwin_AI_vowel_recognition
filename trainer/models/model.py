import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)


class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(6,12, kernel_size=3, stride=1, padding=1) # 20x20x6 -> 20x20x12
        self.bn1 = nn.BatchNorm2d(12) # 20x20x12
        self.relu1 = nn.ReLU() 
        self.pool1 = nn.MaxPool2d(2, 2) # 20x20x12 -> 10x10x12

        self.conv2 = nn.Conv2d(12,24, kernel_size=3, stride=1, padding=1) # 10x10x12 -> 10x10x24
        self.bn2 = nn.BatchNorm2d(24) # 10x10x24
        self.relu2 = nn.ReLU() 
        self.pool2 = nn.MaxPool2d(2, 2) # 10x10x24 -> 5x5x24

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 2x2x48 -> 1x1x48
        self.linear = nn.Linear(24, num_classes) # 1x1x24 -> 1x1x2
        self.softmax = nn.Softmax(dim=1)
        logger.debug("CNN model created")
        logger.debug("CNN model: %s" % self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
#         x = F.dropout(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
#         x = F.dropout(x)
        x = self.pool2(x)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.softmax(x)

        return x



