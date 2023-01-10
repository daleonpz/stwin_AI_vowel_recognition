import logging
import torch
import torch.utils.data

logger = logging.getLogger(__name__)

# split custom dataset into train, test, validation with 80, 10, 10 split
def split_dataset( Dataset ):
    train_size = int(0.8 * len(Dataset))
    test_size = int(0.1 * len(Dataset))
    val_size = len(Dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(Dataset, [train_size, test_size, val_size])
    logger.debug("train size: %d, test size: %d, val size: %d", train_size, test_size, val_size)

    return train_dataset, test_dataset, val_dataset



