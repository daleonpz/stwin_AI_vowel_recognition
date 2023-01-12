from modules.dataset import CustomDataset
from modules.train   import *
from modules.utils   import *
from models.cnn_2    import CNN

import argparse
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import os 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(model, test_loader, device, criterion):
    test_loss, test_acc = validate(model, 
              criterion=criterion,
              val_loader=test_loader, 
              device=DEVICE)
    print(f'test: acc {test_acc} \t  loss {test_loss}')

def show_confusion_matrix(model, test_loader, device, criterion):
    y_pred = []
    y_true = []

    with torch.no_grad():
        for data in test_loader:
            test_data, test_labels = (
                data[0].to(DEVICE, dtype=torch.float32),
                data[1].to(DEVICE),
            )

        pred = model(test_data)
        print(f'pred: {100*pred.float()}')
        pred = pred.argmax(dim=1)
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())

    print(f'true_labels {y_true}')
    print(f'pred_labels {y_pred}')
    print(classification_report(y_true, y_pred,  digits=3))

    cm  = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    plt.show()
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.savefig('results/confusion_matrix.png')

def loss_plot(title):
    plt.figure(figsize=(10,5))
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    
def acc_plot(title):
    plt.figure(figsize=(10,5))
    plt.title(title)
    plt.xlabel("Epochs")
    plt.axhline(y=90, color='red', label="Acceptable accuracy")
    plt.ylabel("Accuracy")

def plot_results(dict_log, epochs, len):
    x_val = np.arange(1, epochs+1)

    loss_plot('Losses per Epoch')
    plt.scatter(x_val, dict_log["val_loss_hist"], label='Validation loss')
    plt.scatter(x_val, dict_log["train_loss_hist"], label='Train loss')
    plt.legend()
    plt.show()
    plt.savefig('results/losses.png')

    plt.figure(figsize=(14,8))
    acc_plot('Validation Accuracy per epoch')
    plt.scatter(x_val, dict_log["val_acc_hist"], label="Validation acc")
    plt.scatter(x_val, dict_log["train_acc_hist"], label='Train acc')
    plt.legend()
    plt.show()
    plt.savefig('results/acc.png')

def main(dataset_path, num_epochs=10, batch_size=16, learning_rate=0.00001):
    logging.getLogger('models.cnn_2').setLevel(logging.INFO)
    logging.getLogger('modules.dataset').setLevel(logging.INFO)
    logging.getLogger('modules.train').setLevel(logging.INFO)

    labels_map= {"A": 0, "E": 1}

    ## Pre tranining config
    dataset = CustomDataset(dataset_path, labels_map)

    model     = CNN(fc_num_output=2, fc_hidden_size=[8]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_ds, val_ds, test_ds = split_dataset(dataset, [0.6,0.3,0.1])

    train_loader  = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    val_loader    = DataLoader(val_ds,   batch_size = batch_size, shuffle=False)
    test_loader   = DataLoader(test_ds,  batch_size = batch_size, shuffle=False)


    dict_log = train(model, 
                     optimizer = optimizer,
                     criterion = criterion,
                     num_epochs = num_epochs,
                     train_loader = train_loader, 
                     val_loader = val_loader,
                     device = DEVICE)


    os.makedirs('results', exist_ok=True)
    plot_results(dict_log, num_epochs, len(train_loader))

    test(model, test_loader, DEVICE, criterion)

    show_confusion_matrix(model, test_loader, DEVICE, criterion)

    torch.save(model.state_dict(), 'results/model.pth')

if __name__ == '__main__':
    print(f'Using device: {DEVICE}')

    ## read dataset_path, num_epochs, batch_size, learning_rate from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data/processed')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.00001)

    print(parser.parse_args())
    args = parser.parse_args()
    main(args.dataset_path, args.num_epochs, args.batch_size, args.learning_rate)

