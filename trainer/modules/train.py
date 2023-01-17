from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

# from models.cnn      import CNN
# from models.cnn_2    import CNN
from models.model    import CNN

logger = logging.getLogger(__name__)

## This two functions will be used everywhere
def get_labels_map():
    labels_map= {"A": 0, "E": 1, "I": 2, "O": 3, "U": 4}
    return labels_map

def get_model(num_classes, device):
#     model     = CNN(fc_num_output=num_classes).to(device) # from cnn, most complicated model -> performs well
    model   = CNN(num_classes).to(device) # from model with global average pooling
    logger.info(f'Defined Model: {model}')
    return model
###########################################################

def train_one_epoch(model, optimizer, criterion, train_loader, device):
    loss_step, metric_step = [], []
    model.train()
    correct_predictions, total_predictions = 0, 0

    for inp_data, labels in train_loader:
        # zero the parameter gradients
        optimizer.zero_grad()
        
        labels      = labels.view(labels.shape[0]).to(device)
        # why use torch.tensor?
        #           ERROR MESSAGE:
        #                   Input type (double) and bias type (float) should be the same
        #           DEBUG:modules.train:input shape: torch.Size([20, 20, 20, 6])
        #           DEBUG:modules.train:     input type: torch.float64
        # convert to float32
        inp_data    = inp_data.to(device, dtype=torch.float32)
        logger.debug(f'label shape: {labels.shape}')
        logger.debug(f'\t label type: {type(labels)}')

        logger.debug(f'input shape: {inp_data.shape}')
        logger.debug(f'\t input type: {inp_data.dtype}')
        outputs     = model(inp_data)

        logger.debug(f'output shape: {outputs.shape}')
        logger.debug(f'\t output type: {outputs.dtype}')

        loss        = criterion(outputs, labels)

        loss_step.append(loss.item())

        loss.backward()
        optimizer.step()
    
        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            total_predictions   += labels.size(0)
            correct_predictions += (predicted == labels).sum()

    loss_curr_epoch = torch.tensor(loss_step).mean().numpy()
    train_acc = 100*correct_predictions / total_predictions

    return loss_curr_epoch, train_acc


def validate(model, criterion, val_loader, device):
    model.eval()
    correct_predictions, total_predictions = 0, 0
    loss_step = []

    with torch.no_grad():
        for inp_data, labels in val_loader:
            labels      = labels.view(labels.shape[0]).to(device)
            inp_data    = inp_data.to(device, dtype=torch.float32)

            outputs = model(inp_data)

            val_loss = criterion(outputs, labels)
            predicted = torch.max(outputs, 1)[1]

            total_predictions   += labels.size(0)
            correct_predictions += (predicted == labels).sum()

            loss_step.append(val_loss.item())

        val_loss_epoch = torch.tensor(loss_step).mean().numpy()
        val_acc = 100*correct_predictions / total_predictions


        return val_loss_epoch, val_acc



def train(model, optimizer, criterion, num_epochs, train_loader, val_loader, device):
#     best_val_loss   = 1000
#     best_val_acc    = 0

    model = model.to(device)

    dict_log = {"train_acc_hist":[], "val_acc_hist":[], "train_loss_hist":[], "val_loss_hist":[]}

    _, train_acc    = validate(model, criterion, train_loader, device)
    _, val_acc      = validate(model, criterion, val_loader, device)

    logger.info(f'Init Accuracy of the model: Train:{train_acc:.3f} \t Val:{val_acc:3f}')


    progress_bar = tqdm(range(num_epochs))

    for epoch in progress_bar:
        train_loss, train_acc  = train_one_epoch(model, optimizer, criterion, train_loader, device)
        val_loss, val_acc      = validate(model, criterion, val_loader, device)

        # Print epoch results to screen 
        msg = (f'Ep {epoch}/{num_epochs}: Accuracy : Train:{train_acc:.2f} \t Val:{val_acc:.2f} || Loss: Train {train_loss:.3f} \t Val {val_loss:.3f}')
        progress_bar.set_description(msg)

        # Track stats
        dict_log["train_acc_hist"].append(train_acc)
        dict_log["val_acc_hist"].append(val_acc)
        dict_log["train_loss_hist"].append(train_loss)
        dict_log["val_loss_hist"].append(val_loss)

    return dict_log

