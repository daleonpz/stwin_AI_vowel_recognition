# train a custom model, activation softmax, loss cross entropy, optimizer adam, learning rate 0.001, batch size 32, epoch 10 in pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)

def train_one_epoch(model, optimizer, train_loader, device, criterion):
    loss_step, metric_step = [], []
    model.train()
    correct_predictions, total_predictions = 0, 0

    for inp_data, labels in train_loader:
        labels      = labels.view(labels.shape[0]).to(device)
        inp_data    = inp_data.to(device)
        logger.debug(f'label shape: {labels.shape}')
        logger.debug(f'\t label type: {type(labels)}')

        logger.debug(f'input shape: {inp_data.shape}')
        logger.debug(f'\t input type: {inp_data.dtype}')
        outputs     = model(inp_data)

        logger.debug(f'output shape: {outputs.shape}')
        logger.debug(f'\t output type: {outputs.dtype}')

        loss        = criterion(outputs, labels)

        loss_step.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            total_predictions   += labels.size(0)
            correct_predictions += (predicted == labels).sum()

    loss_curr_epoch = torch.tensor(loss_step).mean().numpy()
    train_acc = 100*correct_predictions / total_predictions

    return loss_curr_epoch, train_acc


# split custom dataset into train, test, validation with 80, 10, 10 split
def split_dataset( Dataset ):
    train_size = int(0.8 * len(Dataset))
    test_size = int(0.1 * len(Dataset))
    val_size = len(Dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(Dataset, [train_size, test_size, val_size])
    return train_dataset, test_dataset, val_dataset



