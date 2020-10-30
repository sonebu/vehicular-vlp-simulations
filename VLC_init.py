import numpy as np
import time
import os
import subprocess
import multiprocessing
import resource
import matplotlib.pyplot as plt
import math

from functools import lru_cache

"""
*: coordinate center of cari
|--------|                     
| car1   |                      
|-------*|
         |
       y |                   |---------|
         |                   |  car2   |
         |-------------------|*--------|
                    d
"""


def amplitude_modulation():
    noise = np.random.normal(0, 1, 1000)
    """
        0 is the mean of the normal distribution you are choosing from
        1 is the standard deviation of the normal distribution
        100 is the number of elements you get in array noise
    
        Carrier wave c(t)=A_c*cos(2*pi*f_c*t)
        Modulating wave m(t)=A_m*cos(2*pi*f_m*t)
        Modulated wave s(t)=A_c[1+mu*cos(2*pi*f_m*t)]cos(2*pi*f_c*t)
    """

    A_c = 20  # float(input('Enter carrier amplitude: '))
    f_c = 500000000000000  # float(input('Enter carrier frquency: '))
    A_m = 20  # float(input('Enter message amplitude: '))
    f_m = 40000000  # 40MHz #float(input('Enter message frquency: '))
    modulation_index = 1  # float(input('Enter modulation index: '))

    t = np.linspace(0, 1, 1000)

    carrier = A_c * np.cos(2 * np.pi * f_c * t)
    modulator = A_m * np.cos(2 * np.pi * f_m * t)
    product = A_c * (1 + modulation_index * np.cos(2 * np.pi * f_m * t)) * np.cos(2 * np.pi * f_c * t)

    plt.subplot(3, 1, 1)
    plt.title('Amplitude Modulation')
    plt.plot(modulator, 'g')
    plt.ylabel('Amplitude')
    plt.xlabel('Message signal')

    plt.subplot(3, 1, 2)
    plt.plot(carrier, 'r')
    plt.ylabel('Amplitude')
    plt.xlabel('Carrier signal')

    plt.subplot(3, 1, 3)
    plt.plot(product, color="purple")
    plt.ylabel('Amplitude')
    plt.xlabel('AM signal')

    plt.subplots_adjust(hspace=1)
    plt.rc('font', size=15)
    fig = plt.gcf()
    fig.set_size_inches(16, 9)

    fig.savefig('Amplitude Modulation.png', dpi=100)


class VLC_init:
    def __init__(self):
        self.rxradius = 0.003  # 3mm
        self.lookuptable = {}
        self.alpha, self.theta = 120, 120
        self.distancecar = 1
        self.c = 3 * 10 ** 8  # speed of light(m/s)
        self.trxpos, self.trypos = (-5, -5), (2, 3)  # meter
        self.tx1 = np.array((self.trxpos[0], self.trypos[0]))
        self.tx2 = np.array((self.trxpos[0], self.trypos[0]))
        self.rxxpos, self.rxypos = (0, 0), (0, 1)
        self.rx1 = np.array((self.rxxpos[0], self.rxypos[0]))
        self.rx2 = np.array((self.rxxpos[1], self.rxypos[1]))

        self.distancebtw11 = np.linalg.norm(self.tx1 - self.rx1)
        self.distancebtw12 = np.linalg.norm(self.tx1 - self.rx2)
        self.distancebtw21 = np.linalg.norm(self.tx2 - self.rx1)
        self.distancebtw22 = np.linalg.norm(self.tx2 - self.rx2)

        self.aoa11 = math.atan((self.tx1[1] - self.rx1[1]) / (self.rx1[0] - self.tx1[0]))
        self.aoa12 = math.atan((self.tx1[1] - self.rx2[1]) / (self.rx2[0] - self.tx1[0]))
        self.aoa21 = math.atan((self.tx2[1] - self.rx1[1]) / (self.rx1[0] - self.tx2[0]))
        self.aoa22 = math.atan((self.tx2[1] - self.rx2[1]) / (self.rx2[0] - self.tx2[0]))
        self.aoas = np.array((self.aoa11, self.aoa12), (self.aoa21, self.aoa22))
        self.eps_a = 0
        self.eps_b = 0
        self.eps_c = 0
        self.eps_d = 0

        self.delays = (self.distancebtw11 / self.c, self.distancebtw12 / self.c, self.distancebtw21 / self.c,
                       self.distancebtw22 / self.c)
        self.distances = np.array((self.distancebtw11, self.distancebtw12), (self.distancebtw21, self.distancebtw22))
        self.R1 = np.array([[0, 0], [0, 0]])
        self.area1 = np.array([[0, 0], [0, 0]])
        self.area2 = np.array([[0, 0], [0, 0]])
        self.H = np.array([[0, 0], [0, 0]])

    @lru_cache(maxsize=None)
    def update_cords(self, tx_cord, rx_cord, rx_radius):
        self.rxradius = rx_radius  # mm
        self.trxpos, self.trypos = (tx_cord[0][0], tx_cord[1][0]), (tx_cord[0][1], tx_cord[1][1])  # meter
        self.tx1 = np.array((self.trxpos[0], self.trypos[0]))
        self.tx2 = np.array((self.trxpos[0], self.trypos[0]))
        self.rxxpos, self.rxypos = (rx_cord[0][0], rx_cord[1][0]), (rx_cord[0][1], rx_cord[1][1])
        self.rx1 = np.array((self.rxxpos[0], self.rxypos[0]))
        self.rx2 = np.array((self.rxxpos[1], self.rxypos[1]))

    @lru_cache(maxsize=None)
    def update_lookuptable(self):
        self.calc_delay()
        self.update_aoa()
        self.update_eps()
        self.distances = np.array([[self.distancebtw11, self.distancebtw12], [self.distancebtw21, self.distancebtw22]])
        self.R1 = np.array([[0, 0], [0, 0]])
        self.area1 = np.array([[0, 0], [0, 0]])
        self.area2 = np.array([[0, 0], [0, 0]])
        self.H = np.array([[0, 0], [0, 0]])
        for i in range(2):
            for j in range(2):
                self.R1[i][j] = self.distances[i][j] * (1 / math.tan(math.radians(80)))
                self.area1[i][j] = math.pi * self.R1[i][j] ** 2
                self.area2[i][j] = math.pi * self.rxradius ** 2
                self.H[i][j] = self.area1[i][j] / self.area2[i][j]
        return self.H, self.delays

    @lru_cache(maxsize=None)
    def update_aoa(self):
        self.aoa11 = math.atan((self.tx1[1] - self.rx1[1]) / (self.rx1[0] - self.tx1[0]))
        self.aoa12 = math.atan((self.tx1[1] - self.rx2[1]) / (self.rx2[0] - self.tx1[0]))
        self.aoa21 = math.atan((self.tx2[1] - self.rx1[1]) / (self.rx1[0] - self.tx2[0]))
        self.aoa22 = math.atan((self.tx2[1] - self.rx2[1]) / (self.rx2[0] - self.tx2[0]))
        self.aoas = np.array((self.aoa11, self.aoa12), (self.aoa21, self.aoa22))

    @lru_cache(maxsize=None)
    def update_eps(self, aoa):
        self.eps_a = self.translate(aoa, 0, 80, 1 / 4, 0)
        self.eps_c = self.eps_a
        self.eps_b = (1 - 2 * self.eps_a) / 2
        self.eps_d = self.eps_b

    @lru_cache(maxsize=None)
    def calc_delay(self):
        self.distancebtw11 = np.linalg.norm(self.tx1 - self.rx1)
        self.distancebtw12 = np.linalg.norm(self.tx1 - self.rx2)
        self.distancebtw21 = np.linalg.norm(self.tx2 - self.rx1)
        self.distancebtw22 = np.linalg.norm(self.tx2 - self.rx2)
        self.delays = np.array([[self.distancebtw11 / self.c, self.distancebtw12 / self.c],
                                [self.distancebtw21 / self.c, self.distancebtw22 / self.c]])

    @lru_cache(maxsize=None)
    def translate(self, value, leftMin, leftMax, rightMin, rightMax):
        # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin
        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)
        # Convert the 0-1 range into a value in the right range.
        return rightMin + (valueScaled * rightSpan)
