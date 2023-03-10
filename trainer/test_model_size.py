from modules.dataset import CustomDataset
from modules.train   import *
from modules.utils   import *
from models.cnn_2    import CNN


import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.DEBUG)
logging.root.setLevel(logging.NOTSET)

def main():
    logging.getLogger('modules.dataset').setLevel(logging.WARNING)
    logging.getLogger('modules.train').setLevel(logging.INFO)
    logging.getLogger('modules.utils').setLevel(logging.WARNING)

    num_classes = len( get_labels_map())
    model = get_model(num_classes, DEVICE)
    param_size = 0
    buffer_size = 0
    size_datatype_in_bytes = 4 #float32

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    # print number of parameters per layer
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.nelement())

    size_all_kb = (param_size + buffer_size)/ 1024

    if( size_all_kb > 500 ):
        print("\nNAMED PARAMS")
        for name, param in model.named_parameters():
          print("    ", name, "[", type(name), "]", type(param), param.size())

        print("\nNAMED BUFFERS")
        for name, param in model.named_buffers():
          print("    ", name, "[", type(name), "]", type(param), param.size())

        print("\nSTATE DICT (KEYS VALUE PAIRS)")
        for k, v in model.state_dict().items():
          print("        ", "(", k, "=>", v.shape, ")")

        print("WARNING!!!! The model is too big (> 500KB)")
    else:
        print("Model size is OK")
   
    print('Model Size: {:.3f} KB'.format(size_all_kb))


if __name__ == '__main__':
    main()


