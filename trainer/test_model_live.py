import argparse
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import re
import serial
import sys, getopt
import time
from tqdm import tqdm
import torch

from modules.dataset import *
from modules.train   import *
from modules.utils   import *
from models.cnn_2    import CNN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 400
THRESHOLD = 0.8


def get_label(img, model):
    data = torch.from_numpy(img).float().to(DEVICE)
    # add batch dimension
    data = data.unsqueeze(0)
    with torch.no_grad():
        prediction = model(data)
        print('Raw Prediction: ', prediction)
        prediction = (prediction > THRESHOLD).float()
        print('After thresholding: ', prediction)

        if prediction.sum() == 0:
            print('\t \t No prediction')
            return np.array([99]) # no label

        prediction = prediction.argmax(dim=1)

        return prediction


class serialCollector:
    def __init__(self, 
            serialPort = '/dev/ttyACM0', serialBaud = 115200,
            sample_freq = 20):

        self.port = serialPort
        self.baud = serialBaud
        self.sample_freq = sample_freq 

        print('Trying to connect to: ' + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
        try:
            self.serialConnection = serial.Serial(serialPort, serialBaud, timeout=4)
            print('Connected to ' + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
        except:
            print("Failed to connect with " + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
            sys.exit(1)

    def readSerialStart(self):
        raw_data = self.serialConnection.readline().decode('utf-8')
        print('-----------------------------------------------')
        print('Starting collection of data...')

        data_array = np.zeros((NUM_SAMPLES, 6), dtype=np.float32)

        data = []
        while (len(data) != 6 ):
            raw_data = self.serialConnection.readline().decode('utf-8')
            data = [float(s) for s in re.findall(r'-?\d+\.?\d*', raw_data)]
            print("waiting for a good reading...")
            time.sleep(0.1)

        for i in range(0, NUM_SAMPLES):
            for attempt in range(0, 10):
                raw_data = self.serialConnection.readline().decode('utf-8')
                data = [float(s) for s in re.findall(r'-?\d+\.?\d*', raw_data)]
                if (len(data) != 6):
                    print("data error, new try...")
                    continue
                else:
                    data_array[i, :] = data
                    break

        return data_array

    def close(self):
        self.serialConnection.close()
        print('Disconnected...')


def main(port, baud, sample_freq):

    s = serialCollector(port, baud, sample_freq)

    sampling_time = 1.0 / sample_freq

    labels_map= {"A": 0, "E": 1, "I": 2, "O": 3, "U": 4, "-": 99}
    reverse_labels_map = {0: "A", 1: "E", 2: "I", 3: "O", 4: "U", 99: "-"}

    # load model from file pytorch
    model = CNN(fc_num_output=5, fc_hidden_size=[]).to(DEVICE)
    model.load_state_dict(torch.load('results/model.pth'))
    model.eval()

    while True:
        data = s.readSerialStart()
        data_img = data2image(data)
        label = get_label(data_img, model)
        time.sleep(sampling_time)
        if label[0] != 99:
            print('Predicted label: ', reverse_labels_map[label[0].item()])

    s.close()
        
if __name__ == '__main__':
    print(f'Using device: {DEVICE}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default='/dev/ttyACM0', help='serial port')
    parser.add_argument('--baudrate', type=int, default=115200, help='serial baudrate')
    parser.add_argument('--sample_freq', type=int, default=20, help='sample frequency')

    print(parser.parse_args())

    main(parser.parse_args().port,
         parser.parse_args().baudrate, 
         parser.parse_args().sample_freq)


