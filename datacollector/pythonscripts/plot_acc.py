#!/usr/bin/env python

from threading import Thread
import serial
import re
import time
import collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import struct
import sys
# import pandas as pd


class serialPlot:
    def __init__(self, serialPort = '/dev/ttyACM0', serialBaud = 115200, plotLength = 100, dataNumBytes = 1):
        self.port = serialPort
        self.baud = serialBaud
        self.plotMaxLength = plotLength
        self.dataNumBytes = dataNumBytes
        self.rawData = bytearray(dataNumBytes)
        self.data_x = collections.deque([0] * plotLength, maxlen=plotLength)
        self.data_y = collections.deque([0] * plotLength, maxlen=plotLength)
        self.data_z = collections.deque([0] * plotLength, maxlen=plotLength)
        self.isRun = True
        self.isReceiving = False
        self.thread = None
        self.plotTimer = 0
        self.previousTimer = 0
        # self.csvData = []

        print('Trying to connect to: ' + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
        try:
            self.serialConnection = serial.Serial(serialPort, serialBaud, timeout=4)
            print('Connected to ' + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
        except:
            print("Failed to connect with " + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
            sys.exit(1)

    def readSerialStart(self):
        if self.thread == None:
            self.thread = Thread(target=self.backgroundThread)
            self.thread.start()
            # Block till we start receiving values
            while self.isReceiving != True:
                time.sleep(0.1)

    def getSerialData(self, frame, ax,lines_x, lines_y, lines_z, lineValueText, timeText):
        currentTimer = time.perf_counter()
        self.plotTimer = int((currentTimer - self.previousTimer) * 1000)     # the first reading will be erroneous
        self.previousTimer = currentTimer
        timeText.set_text('Plot Interval = ' + str(self.plotTimer) + 'ms')
#         value,  = struct.unpack('f', self.rawData)    # use 'h' for a 2 byte integer
        value = self.rawData
        self.data_x.append(value[0])    # we get the latest data point and append it to our array
        self.data_y.append(value[1])    # we get the latest data point and append it to our array
        self.data_z.append(value[2])    # we get the latest data point and append it to our array
        lines_x.set_data(range(self.plotMaxLength), self.data_x)
        lines_y.set_data(range(self.plotMaxLength), self.data_y)
        lines_z.set_data(range(self.plotMaxLength), self.data_z)
        ax.relim()
        ax.autoscale_view()
#         lineValueText.set_text('[' + lineLabel + '] = ' + str(value[0]))
        # self.csvData.append(self.data[-1])

    def backgroundThread(self):    # retrieve data
        time.sleep(1.0)  # give some buffer time for retrieving data
        self.serialConnection.reset_input_buffer()

        while (self.isRun):
#             self.serialConnection.readinto(self.rawData)
#             self.isReceiving = True

            raw_data = self.serialConnection.readline().decode('utf-8')
            data = [float(s) for s in re.findall(r'-?\d+\.?\d*', raw_data)]
#             data = data[0::3]
#             data = data[0] # just one value
            self.rawData = data
            self.isReceiving = True
            print(self.rawData)

    def close(self):
        self.isRun = False
        self.thread.join()
        self.serialConnection.close()
        print('Disconnected...')
        # df = pd.DataFrame(self.csvData)
        # df.to_csv('/home/rikisenia/Desktop/data.csv')


def main():
    portName = '/dev/ttyACM0'
    baudRate = 115200 
    maxPlotLength = 250
    dataNumBytes = 1        # number of bytes of 1 data point
    s = serialPlot(portName, baudRate, maxPlotLength, dataNumBytes)   # initializes all required variables
    s.readSerialStart()                                               # starts background thread

    # plotting starts below
    pltInterval = 25    # Period at which the plot animation updates [ms]
    xmin = 0
    xmax = maxPlotLength
    ymin = -(1500)
    ymax = 1500
    fig = plt.figure()
#     ax = plt.axes(xlim=(xmin, xmax), ylim=(float(ymin - (ymax - ymin) / 10), float(ymax + (ymax - ymin) / 10)))
    ax = plt.axes(xlim=(xmin, xmax))
    ax.set_title('Analog Read')
    ax.set_xlabel("time")
    ax.set_ylabel("AnalogRead Value")

    timeText = ax.text(0.50, 0.95, '', transform=ax.transAxes)
    lines_x = ax.plot([], [], label="X")[0]
    lines_y = ax.plot([], [], label="Y")[0]
    lines_z = ax.plot([], [], label="Z")[0]
    lineValueText = ax.text(0.50, 0.90, '', transform=ax.transAxes)
    anim = animation.FuncAnimation(fig, s.getSerialData, 
                                   fargs=(ax, lines_x, lines_y, lines_z, lineValueText, timeText), 
                                   interval=pltInterval, blit= False)    # fargs has to be a tuple

    plt.legend(loc="upper left")
    plt.show()

    s.close()


if __name__ == '__main__':
    main()
