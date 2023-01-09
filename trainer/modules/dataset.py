# create a custom csv dataset in pytorch

import torch
from torch.utils.data import Dataset
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.len = self.data.shape[0]

    def __getitem__(self, index):
        return self.data.iloc[index, 0], self.data.iloc[index, 1]

    def __len__(self):
        return self.len




