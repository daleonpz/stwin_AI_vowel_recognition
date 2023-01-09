# create a fc model with x classes and n features with pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import logging

logger = logging.getLogger(__name__)

class FC(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=[128,48], dropout=0):
        super(FC, self).__init__()
        assert len(hidden_size) >= 1 # at least one hidden layer
        layers = nn.ModuleList()
        layer_sizes = [num_features] + hidden_size

        for dim_in, dim_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(dim_in, dim_out, bias=True))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        # the last layer is the output layer, so we don't want to apply ReLU to it
        layers.append(nn.Linear(layer_sizes[-1], num_classes, bias=True))

        self.layers = nn.Sequential(*layers)

        logger.debug("FC model with %d features, %d classes, %d hidden layers, %d hidden units per layer" % (num_features, num_classes, len(hidden_size), hidden_size[0]))
        logger.debug("FC model: %s" % self)

    def forward(self, x):
        x = x.view(x.shape[0], -1) # flatten 
        x = self.layers(x)
        return x



