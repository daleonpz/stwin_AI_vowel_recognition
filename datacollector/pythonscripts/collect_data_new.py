import argparse
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

NUM_SAMPLES = 400

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
        self.ser = SerialData()
    
    def estimate_gravity_vector(self, acc, new_samples):
        samples = np.min([100, new_samples])
        self.gravity_vector = np.mean(acc[-samples:], axis=0)
#         print("gravity vector: ", self.gravity_vector)

    def estimate_velocity(self, acc, new_samples):
        self.estimate_gravity_vector(acc, new_samples)
        dt = 1 / 200
        self.velocity = np.zeros(3)

        # I assume that the board is always parallel to the ground
        self.gravity_vector = np.array([0, 0, 980.0])
        self.velocity = np.sum(acc[-new_samples:], axis=0) * dt - self.gravity_vector * dt * new_samples

    def get_estimate(self):
        return self.position, self.velocity, self.orientation

    def isMoving(self):
        time.sleep(0.05)
        new_samples = self.ser.update()
        acc, gyro  = self.ser.get_last_n_samples(new_samples)
        self.estimate_velocity(acc, new_samples)

        if np.linalg.norm(self.velocity) > 10.0: # 1.0 for normal velocity, 0.1 for average velocity
            print("norm: ", np.linalg.norm(self.velocity))
            return True
        else:
            return False

if __name__ == '__main__':

    # read arguments from terminal file and label
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str,  default='A')
    parser.add_argument('--samples', type=int, default=20)

    args = parser.parse_args()

    current_estimate = CurrentEstimate()

    # create files following this pattern  label_0001.csv, label_0002.csv, ... using a for loop
    for i in tqdm(range(args.samples)):
        # create a new file
        filename = args.label + '_' + str(i).zfill(4) + '.csv'
        print(f'{filename} is being created')

        # wait for the board to be still
        while current_estimate.isMoving():
            pass

        print("Board is still")

        while current_estimate.isMoving() == False:
            pass

        print("Board is moving")


        # when is not moving start reading all the values
        new_samples = 0
        while( new_samples < NUM_SAMPLES):
            new_samples += current_estimate.ser.update()

        acc, gyro = current_estimate.ser.get_last_n_samples(NUM_SAMPLES)
        data = np.concatenate((acc, gyro), axis=1)

        print(f'data shape: {data.shape}')

        with open(filename, 'w') as f:
            np.savetxt(f, data, delimiter=",")
            print(f'{filename} is saved')

