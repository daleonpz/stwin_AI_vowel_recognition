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


def data2image(data):
    len_data = len(data)
    img_size = math.ceil(math.sqrt(len_data))

    if img_size * img_size != len_data:
        logger.warn('not a perfect square, padding with zeros')
        data = np.pad(data, (0, img_size * img_size - len_data), 'constant')

    ndata = normalize_columns_between_0_and_1(data)

    data = ndata.reshape((6, img_size, img_size))

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
                data = pd.read_csv(dataset_path + "/vowel_" + label + "/" + file, header=None)
                logger.debug(f'data shape as vector: {data.shape}')

                data = data2image(data.values)
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
        "E": 1
    }
    rev_labels_map = {v: k for k, v in labels_map.items()}

    dataset_path = "/home/me/Documents/git/mystuff/tinyml/stwin_AI_vocals_detection/data"
    dataset = CustomDataset(dataset_path, labels_map)

    print(f'Number of samples: {len(dataset)}')
    # test
    print(f'First sample:')
    data, label = dataset[0]
    print(f'\t shape: {data.shape}')
    print(f'\t label: {rev_labels_map[label]}')



