import numpy as np
import time
import os
import subprocess
import multiprocessing
import resource
import matplotlib.pyplot as plt
import math

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


class VLC_init:
    def __init__(self):
        self.rxradius = 0.003  # 3mm
        self.lookuptable = {}
        self.alpha, self.theta = 120, 120
        self.distancecar = 1
        self.c = 3*10**8    # speed of light(m/s)
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
        self.delays = (self.distancebtw11 / self.c, self.distancebtw12 / self.c, self.distancebtw21 / self.c, self.distancebtw22 / self.c)
        self.distances = np.array((self.distancebtw11, self.distancebtw12),(self.distancebtw21, self.distancebtw22))
        self.R1 = np.array([[0, 0], [0, 0]])
        self.area1 = np.array([[0, 0], [0, 0]])
        self.area2 = np.array([[0, 0], [0, 0]])
        self.H = np.array([[0, 0], [0, 0]])


    def update_params(self, tx_cord, rx_cord, rx_radius):
        self.rxradius =  rx_radius # mm
        self.trxpos, self.trypos = (tx_cord[0][0], tx_cord[1][0]), (tx_cord[0][1], tx_cord[1][1])  # meter
        self.tx1 = np.array((self.trxpos[0], self.trypos[0]))
        self.tx2 = np.array((self.trxpos[0], self.trypos[0]))
        self.rxxpos, self.rxypos = (rx_cord[0][0], rx_cord[1][0]), (rx_cord[0][1], rx_cord[1][1])
        self.rx1 = np.array((self.rxxpos[0], self.rxypos[0]))
        self.rx2 = np.array((self.rxxpos[1], self.rxypos[1]))

    def update_lookuptable(self):
        self.calc_delay()
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

    def calc_delay(self):
        self.distancebtw11 = np.linalg.norm(self.tx1 - self.rx1)
        self.distancebtw12 = np.linalg.norm(self.tx1 - self.rx2)
        self.distancebtw21 = np.linalg.norm(self.tx2 - self.rx1)
        self.distancebtw22 = np.linalg.norm(self.tx2 - self.rx2)
        self.delays = np.array([[self.distancebtw11 / self.c, self.distancebtw12 / self.c],
                                [self.distancebtw21 / self.c, self.distancebtw22 / self.c]])


