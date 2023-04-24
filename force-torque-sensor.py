# Get_W - ati_u6_pub without ROS
# !/usr/bin/env python3
from datetime import datetime
from LabJackPython import NullHandleException
import u6

import time
import numpy

# MAX_REQUESTS is the number of packets to be read.
MAX_REQUESTS = 75
# SCAN_FREQUENCY is the scan frequency of stream mode in Hz
SCAN_FREQUENCY = 5000
DAQ_STATE = True  # make sure daq is connected


# calibration_flag = 0 #for first time run
# ref_weight=10 #arbitrary high start point

class ATI_readings:
    # using labjack u6

    def __init__(self, **kwargs):
        self.resolutionIndex = kwargs['resolutionIndex']
        self.gainIndex = kwargs['gainIndex']
        self.settlingFactor = kwargs['settlingFactor']
        self.differential = kwargs['differential']
        self.daq_device = u6.U6()
        self.daq_device.getCalibrationData()
        self.rawData = []
        self.forces = []
        self.calibration_flag = 0  # for ref
        self.ref_weight = 10  # arbitrary high start point
        self.res_weight = 0
        self.ref_res_weight = 10
        self.weight = 0
        self.margin = 0.1  # current threshold: 10grams
        self.GR_flag = 0  # to cehck if we are grabbing or releasing
        self.bias = [-0.28026725005202024, -0.5628451798374954, 0.36969605272634, -0.021619200849272602, -0.1790193724709752, 0.2219047680659969]

    def __str__(self):
        printOut = '''
        Resolution Index: {} 
        Gain Index: {} 
        Settlingfactor: {} 
        Differential: {}
        '''

        return printOut.format(self.resolutionIndex, self.gainIndex, self.settlingFactor, self.differential)

    def getAnalogChannels(self):

        self.isConnected()
        # Read even channels for differential voltages
        channel0 = self.daq_device.getAIN(0)
        channel1 = self.daq_device.getAIN(2)
        channel2 = self.daq_device.getAIN(4)
        channel3 = self.daq_device.getAIN(6)
        channel4 = self.daq_device.getAIN(8)
        channel5 = self.daq_device.getAIN(10)
        self.rawData = [channel0, channel1, channel2, channel3, channel4, channel5]
        # print(self.rawData)


    def convertingRawData(self):
        # from FTxx.cal available on ATI website
        # FT35016
        userAxis = [[-1.54898,   0.18846,   5.89676, -48.38977,  -3.04654, 46.30090],
                    [-7.14653,  55.02803,   0.49310, -28.03857,   4.42144, -26.78444],
                    [69.40862,   1.66791,  69.83037,   2.80683,  65.97221,   2.30815],
                    [-0.06111,   0.38902,  -1.11337,  -0.24520,   1.12728,  -0.14797],
                    [1.27340,  0.04066,  -0.69535,   0.31133,  -0.58715,  -0.34969],
                    [0.10451,  -0.70844,   0.08499,  -0.71084,   0.03483,  -0.68398]]


        offSetCorrection = self.rawData - numpy.transpose(self.bias)
        self.forces = numpy.dot(userAxis, numpy.transpose(offSetCorrection))


    def isConnected(self):
        if self.daq_device is None:
            global DAQ_STATE
            DAQ_STATE = False
            raise NullHandleException()

    def get_weight(self): 

        if self.calibration_flag == 0:
            raise Exception("F/T sensor is not calibrated. Please calibrate it first!")

        checks = 0
        iteration = 10
        Ws = numpy.zeros(iteration)
        Res_Ws = numpy.zeros(iteration)
        

        while checks < iteration:

            if not DAQ_STATE:  # failsafe
                print('DAQ is disconnected!')
                break

            self.getAnalogChannels()  # daq
            self.convertingRawData()  # convert
            Ws[checks] = self.forces[2]  # z output
            Res_Ws[checks] = numpy.sqrt((self.forces[0]) ** 2 + (self.forces[1]) ** 2 + (self.forces[2]) ** 2)
      

            checks += 1

        self.res_weight = numpy.mean(Res_Ws)
        self.weight = numpy.mean(Ws)



    def calibration(self):
        checks = 0
        rawData_array = numpy.zeros((100,6))
        while checks < 100:
            self.getAnalogChannels()
            rawData_array[checks] = self.rawData
            checks += 1
        self.bias = numpy.mean(rawData_array, axis=0)
        self.calibration_flag = 1
        print("Calibration is done!")
        print("Bias is: ", self.bias)


"""
Visualization
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt


class RealtimePlot1D():
    def __init__(
            self,
            x_tick,
            length,
            xlabel="Time",
            title="RealtimePlot1D",
            label=None,
            color="c",
            marker='-o',
            alpha=1.0,
            ylim=None
    ):
        self.x_tick = x_tick
        self.length = length
        self.color = color
        self.marker = marker
        self.alpha = 1.0
        self.ylim = ylim
        self.label = label
        self.xlabel = xlabel
        self.title = title
        self.line = None

        # プロット初期化
        self.init_plot()

    def init_plot(self):
        self.x_vec = np.arange(0, self.length) * self.x_tick \
                     - self.length * self.x_tick
        self.y_vec = np.zeros(self.length)

        plt.ion()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        self.line = ax.plot(self.x_vec, self.y_vec,
                            self.marker, color=self.color,
                            alpha=self.alpha)

        if self.ylim is not None:
            plt.ylim(self.ylim[0], self.ylim[1])
        plt.xlabel(self.xlabel)
        plt.title(self.title)
        plt.grid()
        plt.show()

        self.index = 0
        self.x_data = -self.x_tick
        self.pretime = 0.0
        self.fps = 0.0

    def update_index(self):
        self.index = self.index + 1 if self.index < self.length - 1 else 0

    def update_ylim(self, y_data):
        ylim = self.line[0].axes.get_ylim()
        if y_data < ylim[0]:
            plt.ylim(y_data * 1.1, ylim[1])
        elif y_data > ylim[1]:
            plt.ylim(ylim[0], y_data * 1.1)

    def compute_fps(self):
        curtime = time.time()
        time_diff = curtime - self.pretime
        self.fps = 1.0 / (time_diff + 1e-16)
        self.pretime = curtime

    def update(self, y_data):
        # プロットする配列の更新
        self.x_data += self.x_tick
        self.y_vec[self.index] = y_data

        y_pos = self.index + 1 if self.index < self.length else 0
        tmp_y_vec = np.r_[self.y_vec[y_pos:self.length], self.y_vec[0:y_pos]]
        self.line[0].set_ydata(tmp_y_vec)
        if self.ylim is None:
            self.update_ylim(y_data)

        plt.title(f"fps: {self.fps:0.1f} Hz")
        plt.pause(0.01)

        # 次のプロット更新のための処理
        self.update_index()
        self.compute_fps()



"""
Execute
"""
if __name__ == '__main__':
    # ft setup
    ati_ft = ATI_readings(resolutionIndex=1, gainIndex=0, settlingFactor=0, differential=True)
    print("Checking F/T sensor:")
    print(ati_ft.__str__())

    ati_ft.calibration()  # output

    # Graph
    x_tick = 1  # 時間方向の間隔
    length = 200  # プロットする配列の要素数
    realtime_plot1d = RealtimePlot1D(x_tick, length)

    g = 9.80665

    while True:
        #is_weighted, weight = ati_ft.weight_check()
        #print(f'Is something in grasp: {is_weighted}\n')
        ati_ft.get_weight()


        #print(f'average weight detected: {weight * 100}g\n')
        print(f'x {ati_ft.forces[0]}N')
        print(f'y {ati_ft.forces[1]}N')
        print(f'z {ati_ft.forces[2]}N')
        print(f'mx {ati_ft.forces[3]}Nm')
        print(f'my {ati_ft.forces[4]}Nm')
        print(f'mz {ati_ft.forces[5]}Nm')        

        weight = -ati_ft.forces[2] / g * 1000  # g
        print(f'z_weight {weight}g')



        realtime_plot1d.update(weight)

