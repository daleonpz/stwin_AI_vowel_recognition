import collections
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import re
import serial
import time


def normalize_between_0_and_1(data):
    flat_data = data.flatten()

    mmin_data = np.min(flat_data)
    mmax_data = np.max(flat_data)

    data = data - mmin_data
    data = data / (mmax_data - mmin_data)

    return data

# plot two images side by side in a figure as subplots
def plot_two_images_side_by_side(image1, image2, title1, title2):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image1)
    ax2.imshow(image2)
    ax1.set_title(title1)
    ax2.set_title(title2)
    plt.show(block=False)


# plot two vectors side by side in a figure as subplots
def plot_two_vectors_side_by_side(vector1, vector2, title1, title2):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(vector1)
    ax2.plot(vector2)
    ax1.set_title(title1)
    ax2.set_title(title2)
    plt.show(block=False)


# plot data in 2x3 subplots
def plot_data_lin(macc, linacc, mgyro, lingyro):
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(macc)
    ax[0, 1].plot(linacc)
    ax[1, 0].plot(mgyro)
    ax[1, 1].plot(lingyro)

    plt.show()


# plot data in 2x3 subplots
def plot_data(macc, linacc, mgyro, lingyro):
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
    ax1.imshow(macc[:, :, 0])
    ax2.imshow(macc[:, :, 1])
    ax3.imshow(macc[:, :, 2])
    ax4.plot(linacc)
    
    ax5.imshow(mgyro[:, :, 0])
    ax6.imshow(mgyro[:, :, 1])
    ax7.imshow(mgyro[:, :, 2])
    ax8.plot(lingyro)

    plt.show()

    # show plot fullscreen
#     mng = plt.get_current_fig_manager()
#     mng.full_screen_toggle()
#     mng.resize(*mng.window.maxsize())

#     plt.show(block=False)

# get moving average of a vector
def get_moving_average(vector, window_size):
    return np.convolve(vector, np.ones((window_size,)) / window_size, mode="valid")


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
# 
#     class __Full:
#         def append(self, x):
#             self.acc[self.cur] = x[0:3]
#             self.gyro[self.cur] = x[3:6]
#             self.cur = (self.cur + 1) % self.size_max
# 
#         def get(self):
#             """return the buffer in chronological order"""
#             return (
#                 self.acc[self.cur :] + self.acc[: self.cur],
#                 self.gyro[self.cur :] + self.gyro[: self.cur],
#             )
# 
#     def append(self, x):
#         self.acc.append(x[0:3])
#         self.gyro.append(x[3:6])
#         if (len(self.gyro) == self.size_max) or (len(self.acc) == self.size_max):
#             self.cur = 0
#             self.__class__ = self.__Full
# 
#     def get(self):
#         return self.acc, self.gyro
# 

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
        return acc[-n:], gyro[-n:]


class CurrentEstimate:
    def __init__(self):
        self.gyro_bias = []
        self.gravity_vector = []
        self.orientation = [0, 0, 0]
        self.velocity = [0, 0, 0]
        self.position = [0.0, 0.0, 0.0]

    def update_gyro_bias(self, gyro):
        self.gyro_bias = gyro

    def estimate_gyro_bias(self, gyro):
        if self.isMoving():
            return
        samples = 20
        # gyros = [oldest, ..., newest]
        self.gyro_bias = np.mean(gyro[-samples:], axis=0)
        print("gyro bias: ", self.gyro_bias)

    def estimate_gravity_vector(self, acc, new_samples):
        samples = np.min([100, new_samples])
        self.gravity_vector = np.mean(acc[-samples:], axis=0)
#         print("gravity vector: ", self.gravity_vector)

    def estimate_orientation(self,  gyro, new_samples):
        dt = 1 / 200

        print("gyro length: ", len(gyro))
        for i in range(1, new_samples):
            self.orientation = self.orientation + dt * (gyro[-i] - self.gyro_bias)
#             self.orientation = normalize(self.orientation)

        print("orientation: ", self.orientation)

    def estimate_velocity(self, acc, new_samples):
        self.estimate_gravity_vector(acc, new_samples)
        dt = 1 / 200
        self.velocity = [0, 0, 0]
        
        # Due to the integral, we need to start at 1

        for i in range(1, new_samples):
            self.velocity = self.velocity + dt * (acc[i] - self.gravity_vector)


        self.velocity /= new_samples # average velocity  is more reliable

#         print(f'v: {self.velocity}')
#         print(f'avg_v: {self.velocity/new_samples}')
#         print(f'norm(avrg_v): {np.linalg.norm(self.velocity)/new_samples}')
#             self.velocity = self.velocity + dt * (acc[-i] - self.gravity_vector)
#             print(f'v: {self.velocity} a: {acc[-i]} g: {self.gravity_vector}')

#         print("velocity: ", self.velocity)
# 
#     def estimate_position(self, acc, new_samples):
#         dt = 1 / 200
#         self.position = [0, 0, 0]
#         for i in range(new_samples):
#             self.position = self.position + dt * self.velocity
#             self.position = self.position + 0.5 * dt * dt * (
#                 acc[-i] - self.gravity_vector
#             )
# 
#         print("position: ", self.position)

    def get_estimate(self):
        return self.position, self.velocity, self.orientation


    def isMoving(self):
        if np.linalg.norm(self.velocity) > 0.1: # 1.0 for normal velocity, 0.1 for average velocity
            print("norm: ", np.linalg.norm(self.velocity))
            return True
        else:
            return False



if __name__ == "__main__":
    ser = SerialData()
    current_estimate = CurrentEstimate()

    sample_freq = 200
    sample_period = 1 / sample_freq
    sampling_time = 1

    n_samples = int(sampling_time * sample_freq)

    position = np.zeros((n_samples, 3))
    sleep_time = 5

    time.sleep(sleep_time)  # to get enough data
    n_samples = 400 # 196 number of samples in 1 second
    img_size = math.ceil(math.sqrt(n_samples))

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

        print("I am moving")

        # measure execution time 
        start_time = time.time()
        new_samples = 0
        while( new_samples < n_samples):
            new_samples += ser.update()
#             print("new samples: ", new_samples)
#             time.sleep(1)

        stop_time = time.time()
        print("time: ", stop_time - start_time)
        print("show measurements")

        acc, gyro = ser.get_last_n_samples(n_samples)
        print("acc shape: ", np.shape(acc))
        print("gyro shape: ", np.shape(gyro))
        acc = np.array(acc)
        gyro = np.array(gyro)

        
#         if len(acc) < n_samples:
#             time.sleep(sleep_time)
#             continue
# 
        nacc = normalize_between_0_and_1(acc)
        ngyro = normalize_between_0_and_1(gyro)

        plot_data_lin(nacc, acc, ngyro, gyro)

        nacc = nacc.reshape(img_size, img_size, 3)
        ngyro = ngyro.reshape(img_size, img_size, 3)
# 
#         plot_data(nacc, acc, ngyro, gyro)
#         plt.pause(sleep_time)
#         plt.close("all")
