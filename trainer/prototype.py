import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import re
import serial
import time
from tqdm import tqdm
import torch
import sys, getopt

from modules.dataset import *
from modules.train   import *
from modules.utils   import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 400
THRESHOLD = 0.8


def get_label(img, model):
    data = torch.from_numpy(img).float().to(DEVICE)
    # add batch dimension
    data = data.unsqueeze(0)
    with torch.no_grad():
        prediction = model(data)
        # print prediction as float with 2 decimals
#         np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
#         np.set_printoptions(suppress=True)
        print('Raw Prediction: ', np.around(prediction,3))
        prediction = (prediction > THRESHOLD).float()
        print('After thresholding: ', prediction)

        if prediction.sum() == 0:
            print('\t \t No prediction')
            return np.array([99]) # no label

        prediction = prediction.argmax(dim=1)

        return prediction


def normalize_between_0_and_1(data):
    flat_data = data.flatten()

    mmin_data = np.min(flat_data)
    mmax_data = np.max(flat_data)

    data = data - mmin_data
    data = data / (mmax_data - mmin_data)

    return data

# plot data in 2x3 subplots
def plot_data_lin(macc, linacc, mgyro, lingyro):
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(macc)
    ax[0, 1].plot(linacc)
    ax[1, 0].plot(mgyro)
    ax[1, 1].plot(lingyro)

    plt.show()

# create a ring buffer class
class RingBuffer:
    def __init__(self, size_max):
        self.gyro = [[0, 0, 0]] * size_max
        self.acc = [[0, 0, 0]] * size_max
        self.size_max = size_max
        self.cur = 0

    def append(self, x):
        self.acc[self.cur] = x[0:3]
        self.gyro[self.cur] = x[3:6]
        self.cur = (self.cur + 1) % self.size_max

    def get(self):
        """return the buffer in chronological order"""
        return (
            self.acc[self.cur :] + self.acc[: self.cur],
            self.gyro[self.cur :] + self.gyro[: self.cur],
        )

# create a class to read the data from the serial port
class SerialData:
    def __init__(self, serial_port="/dev/ttyACM0", serial_baud=115200):
        self.ser = serial.Serial(serial_port, serial_baud)
        self.buffer = RingBuffer(1000)

    def update(self):
        data = []
        buffer = []
        while len(buffer) < 6:
            buffer = self.ser.read(self.ser.inWaiting())

        recent_data = buffer.decode("utf-8").splitlines()

        data = []
        for i in recent_data:
            data = [float(s) for s in re.findall(r"-?\d+\.?\d*", i)]
            if len(data) == 6:
                self.buffer.append(data)
#                 print(data)

        return len(recent_data)

    def get_buffer(self):
        return self.buffer.get()

    def get_last_n_samples(self, n):
        acc, gyro = self.buffer.get()
        acc = np.array(acc)
        gyro = np.array(gyro)

        return acc[-n:], gyro[-n:]


class CurrentEstimate:
    def __init__(self):
        self.gravity_vector = np.array([0.0,0.0,0.0])
        self.velocity = np.array([0, 0, 0])
    
    def estimate_gravity_vector(self, acc, new_samples):
        samples = np.min([100, new_samples])
        self.gravity_vector = np.mean(acc[-samples:], axis=0)
#         print("gravity vector: ", self.gravity_vector)

    def estimate_velocity(self, acc, new_samples):
        self.estimate_gravity_vector(acc, new_samples)
        dt = 1 / 200
        self.velocity = np.zeros(3)

        # Due to the integral, we need to start at 1

        for i in range(1, new_samples):
            self.velocity = self.velocity + dt * (acc[i] - self.gravity_vector)


        self.velocity /= new_samples # average velocity  is more reliable

    def get_estimate(self):
        return self.position, self.velocity, self.orientation


    def isMoving(self):
        if np.linalg.norm(self.velocity) > 0.1: # 1.0 for normal velocity, 0.1 for average velocity
            print("norm: ", np.linalg.norm(self.velocity))
            return True
        else:
            return False



if __name__ == "__main__":
    print(f'Using device: {DEVICE}')
    ser = SerialData()
    current_estimate = CurrentEstimate()

    img_size = math.ceil(math.sqrt(NUM_SAMPLES))

    labels_map = get_labels_map()
    reverse_labels_map = {v: k for k, v in labels_map.items()}

    num_classes = len(labels_map)
    model = get_model(num_classes, DEVICE)
    print(model)
    model.load_state_dict(torch.load('results/model.pth'))
    model.eval()


    print("Rdy? go!")
    while True:
        time.sleep(0.05)
        new_samples = ser.update()
#         print("new samples: ", new_samples)
        acc, gyro = ser.get_last_n_samples(new_samples)

        current_estimate.estimate_velocity(acc, new_samples)
#         print("acc shape: ", np.shape(acc))
        if current_estimate.isMoving() == False:
            continue
        print('--------------------------------------')
        print("I am moving")

        # measure execution time 
        start_time = time.time()
        new_samples = 0
        while( new_samples < NUM_SAMPLES):
            new_samples += ser.update()
#             print("new samples: ", new_samples)
#             time.sleep(1)

        stop_time = time.time()
        print("time: ", stop_time - start_time)
#         print("show measurements")

        acc, gyro = ser.get_last_n_samples(NUM_SAMPLES)
#         print("acc shape: ", np.shape(acc))
#         print("gyro shape: ", np.shape(gyro))

        nacc = normalize_between_0_and_1(acc)
        ngyro = normalize_between_0_and_1(gyro)

#         plot_data_lin(nacc, acc, ngyro, gyro)

        data = np.concatenate((nacc, ngyro), axis=1)
        data_img = data2image(data)
        label = get_label(data_img, model)
        if label[0] != 99:
            print('Predicted label: ', reverse_labels_map[label[0].item()])

