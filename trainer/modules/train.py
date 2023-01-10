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

        # why use torch.tensor?
        #           ERROR MESSAGE:
        #                   Input type (double) and bias type (float) should be the same
        #           DEBUG:modules.train:input shape: torch.Size([20, 20, 20, 6])
        #           DEBUG:modules.train:     input type: torch.float64
        # convert to float32
        inp_data    = torch.tensor(inp_data, dtype=torch.float32).to(device)
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

