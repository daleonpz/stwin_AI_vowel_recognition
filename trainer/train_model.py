from modules.dataset import * 
from modules.train   import *
from modules.utils   import *


import argparse
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import mlflow

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
#     torch.quantization.fuse_modules(model,
#                                     [['conv1','relu1'], 
#                                      ['conv2','relu2']],
#                                     inplace=True)
#     torch.quantization.fuse_modules(model,
#                                     [['conv1', 'bn1','relu1'], 
#                                      ['conv2', 'bn2','relu2']],
#                                     inplace=True)
# 
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

    return test_loss, test_acc

def get_confusion_matrix(model, test_loader, device, criterion):
    y_pred = []
    y_true = []

    with torch.no_grad():
        for data in test_loader:
            test_data, test_labels = (
                data[0].to(DEVICE, dtype=torch.float32),
                data[1].to(DEVICE),
            )
            # save tensor to csv separated by comma
            print(test_data.shape)
            print(test_labels.shape)
            pred = model(test_data)
            pred = pred.argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())

#                 if test_labels[i].item() == pred[i].item():
#                     _temp = test_data[i].cpu().numpy()
#                     _temp = np.reshape(_temp, (400, 6))
#                     np.savetxt(f'correct_{test_labels[i].item()}.csv', _temp, delimiter=',')

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

def log_results(model, num_epochs, batch_size, learning_rate, optimizer, criterion, train_size, test_size, val_size, test_loss, test_acc, cm, dict_log):
    with mlflow.start_run():
        mlflow.pytorch.log_model(model, "model")

        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("test_size", test_size) 
        mlflow.log_param("val_size", val_size) 

        train_loss = dict_log['train_loss_hist']
        train_acc  = dict_log['train_acc_hist']
        val_loss   = dict_log['val_loss_hist']
        val_acc    = dict_log['val_acc_hist']

        for i in range(len(train_loss)):
            mlflow.log_metric("train_loss", train_loss[i], step=i)
            mlflow.log_metric("train_acc", train_acc[i], step=i)
            mlflow.log_metric("val_loss", val_loss[i], step=i)
            mlflow.log_metric("val_acc", val_acc[i], step=i)

    #     mlflow.log_metric("train_loss", dict_log['train_loss_hist'])
    #     mlflow.log_metric("train_acc", dict_log['train_acc_hist'])
    #     mlflow.log_metric("val_loss", dict_log['val_loss_hist'])
    #     mlflow.log_metric("val_acc", dict_log['val_acc_hist'])

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", test_acc)
#         mlflow.log_param("confusion_matrix", cm)

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
    logging.getLogger('models.cnn').setLevel(logging.INFO)
#     logging.getLogger('models.cnn_2').setLevel(logging.DEBUG)
    logging.getLogger('modules.dataset').setLevel(logging.INFO)
    logging.getLogger('modules.train').setLevel(logging.INFO)
    logging.getLogger('modules.utils').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


    labels_map = get_labels_map()
    NUM_CLASSES = len(labels_map)

    ## Pre tranining config
    dataset = CustomDataset(dataset_path, labels_map)

    model = get_model(NUM_CLASSES, DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_ds, val_ds, test_ds = split_dataset(dataset, [0.7,0.2,0.1])

    train_loader  = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    val_loader    = DataLoader(val_ds,   batch_size = batch_size, shuffle=False)
    test_loader   = DataLoader(test_ds,  batch_size = batch_size, shuffle=False)

#     # print all test samples
#     for data in train_loader:
#         test_data, test_labels = (
#             data[0].to(DEVICE, dtype=torch.float32),
#             data[1].to(DEVICE),
#         )
# 
#     # select 5 random samples
#     fig, ax = plt.subplots(5, 2)
#     for i in range(5):
#         index = np.random.randint(0, len(test_data))
#         _temp = test_data[index].cpu().numpy()
#         _temp = np.reshape(_temp, (400, 6))
# 
#         ax[i, 0].plot(_temp[:,0:3])
#         ax[i, 1].plot(_temp[:,3:6])
# 
# 
#     plt.show()


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

    test_loss, test_acc = test(model, test_loader, DEVICE, criterion)

    log_results(model, 
               num_epochs, 
               batch_size, 
               learning_rate, 
               optimizer, 
               criterion, 
               len(train_ds), len(test_ds), len(val_ds), 
               test_loss, test_acc, 
               cm,
                dict_log)

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

