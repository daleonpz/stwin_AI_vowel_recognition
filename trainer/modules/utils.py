import logging
import torch
import torch.utils.data

logger = logging.getLogger(__name__)

def split_dataset( Dataset, split=[0.8, 0.1, 0.1], shuffle=True, random_seed=42):
    train_size  = int(split[0] * len(Dataset))
    val_size    = int(split[1] * len(Dataset))
    test_size   = len(Dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(Dataset, [train_size, val_size, test_size])

    logger.debug("train size: %d, val size: %d, test size: %d", train_size, val_size, test_size)

    return train_dataset, val_dataset, test_dataset



