# create a pytorch model with two conv2d - relu - maxpool layers

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fc import FC

import logging

logger = logging.getLogger(__name__)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self, fc_num_output = 32, fc_hidden_size=[128,48]).__init__()
        self.conv1 = nn.Conv2d(6, 24, kernel_size=3, stride=1, padding=1) # 3 input channels, 24 output channels, 3x3 kernel
        # equation for output conv layer size: (W-F+2P)/S + 1 # W = input size, F = filter size, P = padding, S = stride
        # (20-3+2)/1 + 1 = 20
        # number of parameters: (3*3*6 + 1)*24 = 1,320
        self.bn1 = nn.BatchNorm2d(24)
        # number of parameters: 24
        self.conv2 = nn.Conv2d(24,48, kernel_size=3, stride=1, padding=1) # 24 input channels, 48 output channels, 3x3 kernel
        # number of parameters: (3*3*24+1)*48 = 10,416
        self.bn2 = nn.BatchNorm2d(48)
        # number of parameters: 48
        self.pool = nn.MaxPool2d(2, 2)  # Downsample the input image by a factor of 2
        # number of parameters: 0
        # equation for output maxpool layer size: (W-F)/S + 1 # W = input size, F = filter size, S = stride
        # (20-2)/2 + 1 = 10

        # In addition the ReLU layers have 24 and 48 parameters each

        # Number of parameters pre fully connected layer: 1,320 + 24 + 24 + 10,416 + 48 + 48 + 0 = 11,880

        # Fully connected layer

        # Number of parameters if two hidden layers [128, 48]
        # 10*10*48 = 4800 
        # 4800*128 + 128 = 614528
        # 128*48 + 48 = 6192
        # 48*2 + 2 = 98
        # Total: 614528 + 6192 + 98 = 620818

#         self.fc = FC(10*10*48, 2) # 10x10x48 input features, 2 output features (for 2 classes)

        self.fc = FC(10*10*48, fc_num_output, fc_hidden_size) 
#         self.fc = FC(10*10*48, 32, [512,128]) # second try 
        logger.debug("CNN model created")
        logger.debug("CNN model: %s" % self)


        # Number of total of parameters: 11,880(CNN) + 620818(FC) = 632968 parameters

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        logger.debug("CNN model: conv1 output shape: %s" % str(x.shape))
        x = F.relu(self.bn2(self.conv2(x)))
        logger.debug("CNN model: conv2 output shape: %s" % str(x.shape))
        x = self.pool(x)
        logger.debug("CNN model: pool output shape: %s" % str(x.shape))
        x = self.fc(x)
#         x = F.log_softmax(x, dim=1) # there is no trainable parameters in this layer
        x = F.Softmax(dim=1)(x) # there is no trainable parameters in this layer

        return x



