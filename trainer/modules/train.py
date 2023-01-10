# train a custom model, activation softmax, loss cross entropy, optimizer adam, learning rate 0.001, batch size 32, epoch 10 in pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train_one_epoch(model, optimizer, train_loader, device, criterion, metric):
    loss_step, metric_step = [], []
    model.train()
    for inp_data, labels in train_loader:
        labels      = labels.view(labels.shape[0]).to(device)
        inp_data    = inp_data.to(device)
        outputs     = model(inp_data)

        loss        = criterion(outputs, labels)
        metric_val  = metric(outputs, labels)

        loss_step.append(loss.item())
        metric_step.append(metric_val.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_curr_epoch = torch.tensor(loss_step).mean().numpy()
    metric_epoch = torch.tensor(metric_step).mean().numpy()
    return loss_curr_epoch, metric_epoch


