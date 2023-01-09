from threading import Thread
import serial
import re
import time
import collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import struct
import sys


class serialPlot:
    def __init__(
        self,
        serialPort="/dev/ttyACM0",
        serialBaud=115200,
        plotLength=100,
        dataNumBytes=1,
    ):
        self.port = serialPort
        self.baud = serialBaud
        self.plotMaxLength = plotLength
        self.dataNumBytes = dataNumBytes
        self.rawData = bytearray(dataNumBytes)

        self.acc_x = collections.deque([0] * plotLength, maxlen=plotLength)
        self.acc_y = collections.deque([0] * plotLength, maxlen=plotLength)
        self.acc_z = collections.deque([0] * plotLength, maxlen=plotLength)

        self.gyro_x = collections.deque([0] * plotLength, maxlen=plotLength)
        self.gyro_y = collections.deque([0] * plotLength, maxlen=plotLength)
        self.gyro_z = collections.deque([0] * plotLength, maxlen=plotLength)

        self.isRun = True
        self.isReceiving = False
        self.thread = None
        self.plotTimer = 0
        self.previousTimer = 0

        print(
            "Trying to connect to: "
            + str(serialPort)
            + " at "
            + str(serialBaud)
            + " BAUD."
        )
        try:
            self.serialConnection = serial.Serial(serialPort, serialBaud, timeout=4)
            print(
                "Connected to " + str(serialPort) + " at " + str(serialBaud) + " BAUD."
            )
        except:
            print(
                "Failed to connect with "
                + str(serialPort)
                + " at "
                + str(serialBaud)
                + " BAUD."
            )
            sys.exit(1)

    def readSerialStart(self):
        if self.thread == None:
            self.thread = Thread(target=self.backgroundThread)
            self.thread.start()
            # Block till we start receiving values
            while self.isReceiving != True:
                time.sleep(0.1)

    def getSerialData(
        self,
        frame,
        ax,
        lines_x_acc,
        lines_y_acc,
        lines_z_acc,
        lines_x_gyro,
        lines_y_gyro,
        lines_z_gyro,
        lineValueText,
        timeText,
    ):
        currentTimer = time.perf_counter()
        self.plotTimer = int(
            (currentTimer - self.previousTimer) * 1000
        )  # the first reading will be erroneous
        self.previousTimer = currentTimer
        timeText.set_text("Plot Interval = " + str(self.plotTimer) + "ms")
        value = self.rawData

        # we get the latest data point and append it to our array
        self.acc_x.append(value[0])
        self.acc_y.append(value[1])
        self.acc_z.append(value[2])

        self.gyro_x.append(value[3])
        self.gyro_y.append(value[4])
        self.gyro_z.append(value[5])

        lines_x_acc.set_data(range(self.plotMaxLength), self.acc_x)
        lines_y_acc.set_data(range(self.plotMaxLength), self.acc_y)
        lines_z_acc.set_data(range(self.plotMaxLength), self.acc_z)

        lines_x_gyro.set_data(range(self.plotMaxLength), self.gyro_x)
        lines_y_gyro.set_data(range(self.plotMaxLength), self.gyro_y)
        lines_z_gyro.set_data(range(self.plotMaxLength), self.gyro_z)

        ax[0].relim()
        ax[0].autoscale_view()

        ax[1].relim()
        ax[1].autoscale_view()

    def backgroundThread(self):  # retrieve data
        time.sleep(1.0)  # give some buffer time for retrieving data
        self.serialConnection.reset_input_buffer()

        while self.isRun:
            raw_data = self.serialConnection.readline().decode("utf-8")
            data = [float(s) for s in re.findall(r"-?\d+\.?\d*", raw_data)]
            self.rawData = data
            self.isReceiving = True
            print(self.rawData)

    def close(self):
        self.isRun = False
        self.thread.join()
        self.serialConnection.close()
        print("Disconnected...")


def main():
    portName = "/dev/ttyACM0"
    baudRate = 115200
    maxPlotLength = 250
    dataNumBytes = 1  # number of bytes of 1 data point
    s = serialPlot(portName, baudRate, maxPlotLength, dataNumBytes)
    s.readSerialStart()  # starts background thread

    # plotting starts below
    pltInterval = 25  # Period at which the plot animation updates [ms]
    xmin = 0
    xmax = maxPlotLength
    ymin = -(1500)
    ymax = 1500

    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].set_title("Accelerometer")
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("Acceleration")
    ax[0].set_xlim(0, maxPlotLength)

    ax[1].set_title("Gyroscope")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("Angular Velocity")
    ax[1].set_xlim(0, maxPlotLength)

    lineValueText = ax[0].text(0.50, 0.95, "", transform=ax[0].transAxes)
    timeText = ax[0].text(0.50, 0.90, "", transform=ax[0].transAxes)

    lines_x_acc = ax[0].plot([], [], label="X")[0]
    lines_y_acc = ax[0].plot([], [], label="Y")[0]
    lines_z_acc = ax[0].plot([], [], label="Z")[0]

    lines_x_gyro = ax[1].plot([], [], label="X")[0]
    lines_y_gyro = ax[1].plot([], [], label="Y")[0]
    lines_z_gyro = ax[1].plot([], [], label="Z")[0]

    anim = animation.FuncAnimation(
        fig,
        s.getSerialData,
        fargs=(
            ax,
            lines_x_acc,
            lines_y_acc,
            lines_z_acc,
            lines_x_gyro,
            lines_y_gyro,
            lines_z_gyro,
            lineValueText,
            timeText,
        ),
        interval=pltInterval,
        blit=False,
    )  # fargs has to be a tuple

    plt.legend(loc="upper left")
    plt.show()
    s.close()


if __name__ == "__main__":
    main()
