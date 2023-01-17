from modules.dataset import * 
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

logging.basicConfig(level=logging.DEBUG)
logging.root.setLevel(logging.NOTSET)

def export_model_to_onnx(model):
    model.eval()
    # Fuse some modules. it may save on memory access, make the model run faster, and improve its accuracy.
    # https://pytorch.org/tutorials/recipes/fuse.html
    torch.quantization.fuse_modules(model,
                                    [['conv1', 'bn1','relu1'], 
                                     ['conv2', 'bn2','relu2']],
                                    inplace=True)

    # Convert to ONNX. 
    # Explanation on why we need a dummy input
    # https://github.com/onnx/tutorials/issues/158
    dummy_input = torch.randn(1, 6, 20, 20) 
    torch.onnx.export(model,
                      dummy_input, 
                      'results/model.onnx', 
                      input_names=['input'], 
                      output_names=['output'])

def test(model, test_loader, device, criterion):
    test_loss, test_acc = validate(model, 
              criterion=criterion,
              val_loader=test_loader, 
              device=DEVICE)
    print(f'Test: acc {test_acc} \t  loss {test_loss}')

def get_confusion_matrix(model, test_loader, device, criterion):
    y_pred = []
    y_true = []


    with torch.no_grad():
        for data in test_loader:
            test_data, test_labels = (
                data[0].to(DEVICE, dtype=torch.float32),
                data[1].to(DEVICE),
            )
            pred = model(test_data)
#             print(f'pred: {100*pred.float()}')
            pred = pred.argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())

    print(f'true_labels {y_true}')
    print(f'pred_labels {y_pred}')
    print(classification_report(y_true, y_pred,  digits=3))

    cm  = confusion_matrix(y_true, y_pred)
    return cm 
    sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    plt.show()
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.savefig('results/confusion_matrix.png')

def loss_plot(title):
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    
def acc_plot(title):
    plt.title(title)
    plt.xlabel("Epochs")
    plt.axhline(y=90, color='red', label="Acceptable accuracy")
    plt.ylabel("Accuracy")

def plot_results(cm, dict_log, epochs, len):
    x_val = np.arange(1, epochs+1)
    
    # subplot 1x3
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(x_val, dict_log['train_loss_hist'], label='train')
    ax1.plot(x_val, dict_log['val_loss_hist'], label='val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.legend()

    ax2.plot(x_val, dict_log['train_acc_hist'], label='train')
    ax2.plot(x_val, dict_log['val_acc_hist'], label='val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    hm = sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues', ax=ax3)
    hm.set_title('Confusion Matrix')
    hm.set_xlabel('Predicted Category')
    hm.set_ylabel('Actual Category')
    fig.tight_layout()
    plt.savefig('results/plot.png')
    plt.show()


def main(dataset_path, num_epochs=10, batch_size=16, learning_rate=0.00001):
    logging.getLogger('models.cnn_2').setLevel(logging.INFO)
    logging.getLogger('modules.dataset').setLevel(logging.INFO)
    logging.getLogger('modules.train').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    labels_map= {"A": 0, "E": 1, "I": 2, "O": 3, "U": 4}

    ## Pre tranining config
    dataset = CustomDataset(dataset_path, labels_map)
    model     = CNN(fc_num_output=5, fc_hidden_size=[]).to(DEVICE)
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

    cm = get_confusion_matrix(model, test_loader, DEVICE, criterion)

    plot_results(cm, dict_log, num_epochs, len(train_loader))

    test(model, test_loader, DEVICE, criterion)

    torch.save(model.state_dict(), 'results/model.pth')

    export_model_to_onnx(model) 

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

