import logging
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd

logger = logging.getLogger(__name__)

def normalize_columns_between_0_and_1(matrix):
    mmin = np.min(matrix, axis=0)
    mmax = np.max(matrix, axis=0)
    matrix = matrix - mmin
    matrix = matrix / (mmax - mmin)
    return matrix

def normalize_acc_gyro(data):
    acc = data[:, 0:3]
    gyro = data[:, 3:6]

    flat_acc = acc.flatten()
    flat_gyro = gyro.flatten()

    mmin_acc = np.min(flat_acc)
    mmax_acc = np.max(flat_acc)
    mmin_gyro = np.min(flat_gyro)
    mmax_gyro = np.max(flat_gyro)

    acc = acc - mmin_acc
    acc = acc / (mmax_acc - mmin_acc)

    gyro = gyro - mmin_gyro
    gyro = gyro / (mmax_gyro - mmin_gyro)

    data[:, 0:3] = acc
    data[:, 3:6] = gyro

    return data

def data2image(data):
    len_data = len(data)
    img_size = math.ceil(math.sqrt(len_data))

    if img_size * img_size != len_data:
        logger.warn('not a perfect square, padding with zeros')
        data = np.pad(data, (0, img_size * img_size - len_data), 'constant')

#     ndata = normalize_columns_between_0_and_1(data)
    ndata = normalize_acc_gyro(data)

    logger.debug(f'first data as normalized vector: {ndata[0, :]}')
    # to ensure that each channel corresponds to a different feature
    data = np.zeros((6, img_size, img_size))
    for j in range(6):
        data[j, :, :] = ndata[:, j].reshape(img_size, img_size)

    assert data.shape == (6, img_size, img_size)
    return data


class CustomDataset(Dataset):
    def __init__(self, dataset_path, labels_map: dict):

        self.data = []
        for label in labels_map.keys():
            files = os.listdir(dataset_path + "/vowel_" + label)
            logger.debug("Number of files in vowel_" + label + ":", len(files))
            
            # iterate through files
            for file in files:
                logger.debug(f'Reading file {file}')
                data = pd.read_csv(dataset_path + "/vowel_" + label + "/" + file, header=None)
                data = data.to_numpy()
                logger.debug(f'data shape as vector: {data.shape}')
                data = data2image(data)

                logger.debug(f'first data as image: {data[:,0,0]}')
                logger.debug(f'data shape as image: {data.shape}')

                self.data.append((data, labels_map[label]))

        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    labels_map = {
        "A": 0,
        "E": 1,
        "I": 2,
        "O": 3,
        "U": 4
    }
    rev_labels_map = {v: k for k, v in labels_map.items()}

    dataset_path = "/home/me/Documents/git/mystuff/tinyml/stwin_AI_vowel_detection/data"
    dataset = CustomDataset(dataset_path, labels_map)

    print(f'Number of samples: {len(dataset)}')
    # test
    print(f'First sample:')
    data, label = dataset[0]
    print(f'\t shape: {data.shape}')
    print(f'\t label: {rev_labels_map[label]}')


