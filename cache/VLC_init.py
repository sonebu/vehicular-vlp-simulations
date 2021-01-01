import math
from functools import lru_cache
import numpy as np
from scipy.integrate import tplquad

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


def diff_volume(p, t, r):
    return r ** 2 * math.sin(p)


def calc_volume(r1, r2, t1, t2, p1, p2):
    # r1, r2: limits for radius (i.e. 0., 1.)
    # t1, t2: limits for theta (i.e. 0, 2*pi)
    # p1, p2: limits for phi (i.e. 0, pi)
    return tplquad(diff_volume, r1, r2, lambda r: t1, lambda r: t2, lambda r, t: p1, lambda r, t: p2)[0]


class VLC_init:
    def __init__(self, rx_radius=0.003, car_dist=1, c=3e8, elavation_angle=80, azimuth_angle=80):
        self.rx_radius = rx_radius
        self.lookuptable = {}
        self.car_dist = car_dist
        self.c = c  # speed of light(m/s)
        self.e_angle, self.a_angle = elavation_angle, azimuth_angle

        self.trxpos, self.trypos = (0, 0), (0, 0)  # meter
        self.tx1 = np.array((0, 0))
        self.tx2 = np.array((0, 0))
        self.rxxpos, self.rxypos = (0, 0), (0, 1)
        self.rx1 = np.array((0, 0))
        self.rx2 = np.array((0, 1))
        self.relative_heading = 0

        self.distancebtw11, self.distancebtw12, self.distancebtw21, self.distancebtw22 = 0, 0, 0, 0

        self.aoa11, self.aoa12, self.aoa21, self.aoa22 = 0, 0, 0, 0
        self.aoas = np.array([[self.aoa11, self.aoa12], [self.aoa21, self.aoa22]])
        self.eps_a, self.eps_b, self.eps_c, self.eps_d, self.phi_h = np.array([[0., 0.], [0., 0.]]), np.array(
            [[0., 0.], [0., 0.]]), np.array([[0., 0.], [0., 0.]]), np.array([[0., 0.], [0., 0.]]), np.array(
            [[0., 0.], [0., 0.]])
        self.delays = np.array([[self.distancebtw11 / self.c, self.distancebtw12 / self.c],
                                [self.distancebtw21 / self.c, self.distancebtw22 / self.c]])
        self.distances = np.array([[self.distancebtw11, self.distancebtw12], [self.distancebtw21, self.distancebtw22]])
        self.H = np.array([[0., 0.], [0., 0.]]).astype(float)

    @lru_cache(maxsize=None)
    def update_coords(self, tx_cord, rx_cord):
        
        self.trxpos, self.trypos = (tx_cord[0][0], tx_cord[1][0]), (tx_cord[0][1], tx_cord[1][1])  # meter
        self.tx1 = np.array((self.trxpos[0], self.trypos[0]))
        self.tx2 = np.array((self.trxpos[1], self.trypos[1]))
        self.rxxpos, self.rxypos = (rx_cord[0][0], rx_cord[1][0]), (rx_cord[0][1], rx_cord[1][1])
        self.rx1 = np.array((self.rxxpos[0], self.rxypos[0]))
        self.rx2 = np.array((self.rxxpos[1], self.rxypos[1]))
        
        self.distancebtw11 = np.linalg.norm(self.tx1 - self.rx1)
        self.distancebtw12 = np.linalg.norm(self.tx1 - self.rx2)
        self.distancebtw21 = np.linalg.norm(self.tx2 - self.rx1)
        self.distancebtw22 = np.linalg.norm(self.tx2 - self.rx2)
        
        self.delays = np.array([[self.distancebtw11 / self.c, self.distancebtw12 / self.c],
                                [self.distancebtw21 / self.c, self.distancebtw22 / self.c]])
        self.distances = np.array([[self.distancebtw11, self.distancebtw12], [self.distancebtw21, self.distancebtw22]])
        
        self.aoa11 = math.atan((self.tx1[1] - self.rx1[1]) / (self.rx1[0] - self.tx1[0]))
        self.aoa12 = math.atan((self.tx1[1] - self.rx2[1]) / (self.rx2[0] - self.tx1[0]))
        self.aoa21 = math.atan((self.tx2[1] - self.rx1[1]) / (self.rx1[0] - self.tx2[0]))
        self.aoa22 = math.atan((self.tx2[1] - self.rx2[1]) / (self.rx2[0] - self.tx2[0]))
        self.aoas = np.array([[self.aoa11, self.aoa12], [self.aoa21, self.aoa22]])
        
        for i in range(2):
            for j in range(2):
                self.eps_a[i][j] = self.translate(self.aoas[i][j], 0, self.e_angle, 1 / 4, 0)
                self.eps_c[i][j] = self.eps_a[i][j]
                self.eps_b[i][j] = (1 - 2 * self.eps_a[i][j]) / 2
                self.eps_d[i][j] = self.eps_b[i][j]
                
        self.H = np.array([[0., 0.], [0., 0.]]).astype(float)
        for i in range(2):
            for j in range(2):
                self.H[i][j] = self.calculate_Hij(i, j)
        

    def calculate_Hij(self, i, j):
        txpos = np.array((self.trxpos[i], self.trypos[i]))
        rxpos = np.array((self.rxxpos[j], self.rxypos[j]))
        distance = self.distances[i][j]
        y = np.abs(txpos[1] - rxpos[1])
        x = np.abs(txpos[0] - txpos[0])
        azimuth = math.atan(((x + self.rx_area * math.cos(self.relative_heading)) / y)) - math.atan(
            ((x - self.rx_area * math.cos(self.relative_heading)) / y))
        elevation = 2 * math.atan((self.rx_area / distance))
        return (elevation / (2 * self.e_angle)) * (azimuth / (2 * self.a_angle))

    @lru_cache(maxsize=None)
    def update_lookuptable(self):
        self.distancebtw11 = np.linalg.norm(self.tx1 - self.rx1)
        self.distancebtw12 = np.linalg.norm(self.tx1 - self.rx2)
        self.distancebtw21 = np.linalg.norm(self.tx2 - self.rx1)
        self.distancebtw22 = np.linalg.norm(self.tx2 - self.rx2)
        self.distances = np.array([[self.distancebtw11, self.distancebtw12], [self.distancebtw21, self.distancebtw22]])
        self.delays = np.array([[self.distancebtw11 / self.c, self.distancebtw12 / self.c],
                                [self.distancebtw21 / self.c, self.distancebtw22 / self.c]])

        self.aoa11 = math.atan((self.tx1[1] - self.rx1[1]) / (self.rx1[0] - self.tx1[0]))
        self.aoa12 = math.atan((self.tx1[1] - self.rx2[1]) / (self.rx2[0] - self.tx1[0]))
        self.aoa21 = math.atan((self.tx2[1] - self.rx1[1]) / (self.rx1[0] - self.tx2[0]))
        self.aoa22 = math.atan((self.tx2[1] - self.rx2[1]) / (self.rx2[0] - self.tx2[0]))
        self.aoas = np.array([[self.aoa11, self.aoa12], [self.aoa21, self.aoa22]])

        for i in range(2):
            for j in range(2):
                self.eps_a[i][j] = self.translate(self.aoas[i][j], 0, self.e_angle, 1 / 4, 0)
                self.eps_c[i][j] = self.eps_a[i][j]
                self.eps_b[i][j] = (1 - 2 * self.eps_a[i][j]) / 2
                self.eps_d[i][j] = self.eps_b[i][j]

        self.H = np.array([[0., 0.], [0., 0.]]).astype(float)
        for i in range(2):
            for j in range(2):
                txpos = np.array((self.trxpos[i], self.trypos[i]))
                rxpos = np.array((self.rxxpos[j], self.rxypos[j]))
                distance = self.distances[i][j]
                y = np.abs(txpos[1] - rxpos[1])
                x = np.abs(txpos[0] - txpos[0])
                azimuth = math.atan(((x + self.rx_area * math.cos(self.relative_heading)) / y)) - math.atan(
                    ((x - self.rx_area * math.cos(self.relative_heading)) / y))
                elevation = 2 * math.atan((self.rx_area / distance))
                self.H[i][j] = (elevation / (2 * self.e_angle)) * (azimuth / (2 * self.a_angle))
        return self.H

    @lru_cache(maxsize=None)
    def update_aoa(self):
        self.aoa11 = math.atan((self.tx1[1] - self.rx1[1]) / (self.rx1[0] - self.tx1[0]))
        self.aoa12 = math.atan((self.tx1[1] - self.rx2[1]) / (self.rx2[0] - self.tx1[0]))
        self.aoa21 = math.atan((self.tx2[1] - self.rx1[1]) / (self.rx1[0] - self.tx2[0]))
        self.aoa22 = math.atan((self.tx2[1] - self.rx2[1]) / (self.rx2[0] - self.tx2[0]))
        self.aoas = np.array([[self.aoa11, self.aoa12], [self.aoa21, self.aoa22]])

    @lru_cache(maxsize=None)
    def update_eps(self):
        for i in range(2):
            for j in range(2):
                self.eps_a[i][j] = self.translate(self.aoas[i][j], 0, self.e_angle, 1 / 4, 0)
                self.eps_c[i][j] = self.eps_a[i][j]
                self.eps_b[i][j] = (1 - 2 * self.eps_a[i][j]) / 2
                self.eps_d[i][j] = self.eps_b[i][j]

    @lru_cache(maxsize=None)
    def calc_delay(self):
        self.distancebtw11 = np.linalg.norm(self.tx1 - self.rx1)
        self.distancebtw12 = np.linalg.norm(self.tx1 - self.rx2)
        self.distancebtw21 = np.linalg.norm(self.tx2 - self.rx1)
        self.distancebtw22 = np.linalg.norm(self.tx2 - self.rx2)
        self.distances = np.array([[self.distancebtw11, self.distancebtw12], [self.distancebtw21, self.distancebtw22]])
        self.delays = np.array([[self.distancebtw11 / self.c, self.distancebtw12 / self.c],
                                [self.distancebtw21 / self.c, self.distancebtw22 / self.c]])
