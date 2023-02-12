import serial
import re
import time
import sys, getopt
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

def normalize_columns_between_0_and_1(matrix):
    mmin = np.min(matrix, axis=0)
    mmax = np.max(matrix, axis=0)
    matrix = matrix - mmin
    matrix = matrix / (mmax - mmin)
    return matrix

def plot_data(mdata, lindata):
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
    ax1.imshow(mdata[:, :, 0])
    ax2.imshow(mdata[:, :, 1])
    ax3.imshow(mdata[:, :, 2])
    ax4.imshow(mdata)

    ax5.plot(lindata[:, 0])
    ax6.plot(lindata[:, 1])
    ax7.plot(lindata[:, 2])

    # show plot fullscreen
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()


class serialCollector:
    def __init__(self, 
            serialPort = '/dev/ttyACM0', serialBaud = 115200,
            sample_freq = 200, sampling_time_sec = 2, 
            csv_filename = 'data.csv', label = 'A'):
        self.port = serialPort
        self.baud = serialBaud
        self.sample_freq = sample_freq 
        self.sampling_time_sec  = sampling_time_sec
        self.csv_filename = csv_filename
        self.label = label

        print('serialCollector:  serialPort = {}, serialBaud = {}, sample_freq = {}, sampling_time_sec = {},  csv_filename = {}, \
                label = {}'.format(serialPort, serialBaud, sample_freq, sampling_time_sec, csv_filename, label))

        print('Trying to connect to: ' + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
        try:
            self.serialConnection = serial.Serial(serialPort, serialBaud, timeout=4)
            print('Connected to ' + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
        except:
            print("Failed to connect with " + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
            sys.exit(1)

    def readSerialStart(self):

        data = []
        while (len(data) != 6 ):
            raw_data = self.serialConnection.readline().decode('utf-8')
            data = [float(s) for s in re.findall(r'-?\d+\.?\d*', raw_data)]
            print("waiting for a good reading...")
        
        with open(self.csv_filename, 'w', newline='') as file:
            print(f'label: {self.label}')
            print('be prepare in 1 sec')
            time.sleep(1)
            print('start now')
            writer = csv.writer(file)
            if isMoving():
                for x in tqdm(range(0, self.sampling_time_sec  * self.sample_freq )):
                    for attempt in range(0, 10):
                        raw_data = self.serialConnection.readline().decode('utf-8')
                        data = [float(s) for s in re.findall(r'-?\d+\.?\d*', raw_data)]
                        if (len(data) != 6):
                            print("data error, new try...")
                            continue
                        else:
                            break

                    writer.writerow(data)

    def close(self):
        self.serialConnection.close()
        print('Disconnected...')


def main(argv):

    portName = '/dev/ttyACM0'
    baudRate = 115200 
    sample_freq = 200
    sampling_time_sec = 2
    csv_filename = "test.csv"
    label = 'A'
    debug = False

    try:
        opts, args = getopt.getopt(argv,"p:b:f:s:F:l:d:",["port=",
             "baudrate=", "sample_freq=", "sample_duration_sec=", "csv_filename=", "label=", "debug="])
    except getopt.GetoptError:
        print('collect_data.py -d <csv_filename>')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print('collect_data.py -d <csv_filename>')
            sys.exit()
        elif opt in ("-p", "--port"):
            portName  = arg
        elif opt in ("-b", "--baudrate"):
            baudrate = arg
        elif opt in ("-f", "--sample_freq"):
            sample_freq = arg
        elif opt in ("-s", "--sample_duration_sec"):
            sampling_time_sec = arg
        elif opt in ("-F", "--csv_filename"):
            csv_filename = arg
        elif opt in ("-l", "--label"):
            label = arg
        elif opt in ("-d", "--debug"):
            debug = arg

    s = serialCollector(portName, baudRate, sample_freq, sampling_time_sec, csv_filename, label)
    s.readSerialStart()
    s.close()

    if debug:
       # read the data from the csv file into a numpy array
        data = np.genfromtxt(csv_filename, delimiter=',')

        acc = data[:, 0:3]
        gyro = data[:, 3:6]

        print(len(acc))
        if len(acc) <  sample_freq * sampling_time_sec:
            print("not enough data")
            return

        nacc = normalize_columns_between_0_and_1(acc)
        ngyro = normalize_columns_between_0_and_1(gyro)

        img_size = math.ceil(math.sqrt(len(acc)))

        print("nacc.shape: ", nacc.shape)
        nacc = nacc.reshape(img_size, img_size, 3)
        print("nacc.shape: ", nacc.shape)

        ngyro = ngyro.reshape(img_size, img_size, 3)
        plot_data(nacc, acc)

if __name__ == '__main__':
    main(sys.argv[1:])
