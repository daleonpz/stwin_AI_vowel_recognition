import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fc import FC

import logging

logger = logging.getLogger(__name__)


class CNN(nn.Module):
    def __init__(self, fc_num_output=2, fc_hidden_size=[32,8]):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12) 
        self.relu1 = nn.ReLU() 
        # output size: 20x20x12

        self.pool1 = nn.MaxPool2d(2, 2)
        # output size: 10x10x12

        self.conv2 = nn.Conv2d(12,24, kernel_size=3, stride=1, padding=1) #
        self.bn2 = nn.BatchNorm2d(24) 
        self.relu2 = nn.ReLU() 
        self.pool2 = nn.MaxPool2d(2, 2) # output size: 5x5x24

        self.fc = FC(5*5*24, fc_num_output, fc_hidden_size) 
        self.softmax = nn.Softmax(dim=1)
        logger.debug("CNN model created")
        logger.debug("CNN model: %s" % self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.fc(x)
        x = self.softmax(x)

        return x



