import serial
import re
import time
import sys, getopt
import csv


class serialCollector:
    def __init__(self, 
            serialPort = '/dev/ttyACM0', serialBaud = 115200,
            sample_freq = 200, sampling_time_sec = 2, 
            csv_filename = 'data.csv'):
        self.port = serialPort
        self.baud = serialBaud
        self.sample_freq = sample_freq 
        self.sampling_time_sec  = sampling_time_sec
        self.csv_filename = csv_filename

        print('Trying to connect to: ' + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
        try:
            self.serialConnection = serial.Serial(serialPort, serialBaud, timeout=4)
            print('Connected to ' + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
        except:
            print("Failed to connect with " + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
            sys.exit(1)

    def readSerialStart(self):
   
        print('Try letter O')
        data = []
        while (len(data) != 3 ):
            raw_data = self.serialConnection.readline().decode('utf-8')
            data = [float(s) for s in re.findall(r'-?\d+\.?\d*', raw_data)]
            print("waiting for a good reading...")
        
        with open(self.csv_filename, 'w', newline='') as file:
            print('be prepare in 0.5 sec')
            time.sleep(0.5)
            print('start now')
            writer = csv.writer(file)
            for x in range(0, self.sampling_time_sec  * self.sample_freq ):
                raw_data = self.serialConnection.readline().decode('utf-8')
                data = [float(s) for s in re.findall(r'-?\d+\.?\d*', raw_data)]
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

    try:
         opts, args = getopt.getopt(argv,"p:b:f:s:F:",["port=",
             "baudrate=", "sample_freq=", "sample_duration_sec=", "csv_filename="])
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


    s = serialCollector(portName, baudRate, sample_freq, sampling_time_sec, csv_filename)
    s.readSerialStart()
    s.close()

if __name__ == '__main__':
    main(sys.argv[1:])
