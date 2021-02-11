import numpy as np
import math

class half_crlb_init:

    def __init__(self, L_1, L_2, rx_area, rx_fov, tx_half_angle, c=3e8):
        """

        :param L_1: distance between two tx leds
        :param L_2: distance between two rx detectors
        :param rx_area: area of one detector
        :param rx_fov: field of view of the receiver detector
        :param tx_half_angle: half angle of the tx led lighting pattern
        :param c: speed of light
        """

        self.L1 = L_1  # m
        self.L2 = L_2  # m
        self.rx_area = rx_area  # m^2

        self.c = c  # speed of light(m/s)
        self.fov = rx_fov  # angle
        self.half_angle = tx_half_angle  # angle
        self.m = -np.log(2) / np.log(math.cos(math.radians(self.half_angle)))

    # derivations for Soner's method
    def d_theta_d_x1(self, ij, tx1, tx2):
        if ij == 11:
            return - tx1[1] / (tx1[0]**2 + tx1[1]**2)
        elif ij == 12:
            return 0
        elif ij == 21:
            return - (tx1[1] - self.L2) / (tx1[0]**2 + (tx1[1] - self.L2)**2)
        elif ij == 22:
            return 0
        else:
            raise ValueError("Entered tx rx values do not exist")


    def d_theta_d_x2(self, ij, tx1, tx2):
        if ij == 11:
            return 0
        elif ij == 12:
            return - tx2[1] / (tx2[0]**2 + tx2[1]**2)
        elif ij == 21:
            return 0
        elif ij == 22:
            return - (tx2[1] - self.L2) / (tx2[0]**2 + (tx2[1] - self.L2)**2)
        else:
            raise ValueError("Entered tx rx values do not exist")


    def d_theta_d_y1(self, ij, tx1, tx2):
        if ij == 11:
            return tx1[0] / (tx1[0 ]**2 + tx1[1 ]**2)
        elif ij == 12:
            return 0
        elif ij == 21:
            return tx1[0] / (tx1[0]**2 + (tx1[1] - self.L2)**2)
        elif ij == 22:
            return 0
        else:
            raise ValueError("Entered tx rx values do not exist")


    def d_theta_d_y2(self, ij, tx1, tx2):
        if ij == 11:
            return 0
        elif ij == 12:
            return tx2[0] / (tx2[0]**2 + tx2[1]**2)
        elif ij == 21:
            return 0
        elif ij == 22:
            return tx2[0] / (tx2[0]**2 + (tx2[1] - self.L2)**2)
        else:
            raise ValueError("Entered tx rx values do not exist")

    # derivations for Roberts' method
    def d_dA_d_x1(self, tx1, tx2):
        return tx1[0] * ( 1. / np.sqrt(tx1[0]**2 + tx1[1]**2) - 1. / np.sqrt(tx1[0]**2 + (tx1[1] + self.L1)**2))


    def d_dA_d_y1(self, tx1, tx2):
        return tx1[1] / np.sqrt(tx1[0]**2 + tx1[1]**2) - (tx1[1] + self.L1) / np.sqrt(tx1[0]**2 + (tx1[1] + self.L1)**2)


    def d_dB_d_x1(self, tx1, tx2):
        return tx1[0] * ( 1. / np.sqrt(tx1[0]**2 + (tx1[1] - self.L2)**2) - 1. / np.sqrt(tx1[0]**2 + (tx1[1] + self.L1 - self.L2)**2))


    def d_dB_d_y1(self, tx1, tx2):
        return tx1[1] / np.sqrt(tx1[0]**2 + (tx1[1] - self.L2)**2) - (tx1[1] + self.L1 - self.L2) / np.sqrt(tx1[0]**2 + (tx1[1] + self.L1 - self.L2)**2)


    # derivations for Bechadeurge's method
    def d_dij_d_x1(self, ij, tx1, tx2):
        if ij == 11:
            return tx1[0] / np.sqrt(tx1[0]**2 + tx1[1]**2)
        elif ij == 12:
            return 0
        elif ij == 21:
            return tx1[0] / np.sqrt(tx1[0]**2 + (tx1[1] - self.L2)**2)
        elif ij == 22:
            return 0
        else:
            raise ValueError("Entered tx rx values do not exist")


    def d_dij_d_x2(self, ij, tx1, tx2):
        if ij == 11:
            return 0
        elif ij == 12:
            return tx2[0] / np.sqrt(tx2[0]**2 + tx2[1]**2)
        elif ij == 21:
            return 0
        elif ij == 22:
            return tx2[0] / np.sqrt(tx2[0]**2 + (tx2[1] - self.L2)**2)
        else:
            raise ValueError("Entered tx rx values do not exist")


    def d_dij_d_y1(self, ij, tx1, tx2):
        if ij == 11:
            return tx1[1] / np.sqrt(tx1[0]**2 + tx1[1]**2)
        elif ij == 12:
            return 0
        elif ij == 21:
            return (tx1[1] - self.L2) / np.sqrt(tx1[0]**2 + (tx1[1] - self.L2)**2)
        elif ij == 22:
            return 0
        else:
            raise ValueError("Entered tx rx values do not exist")


    def d_dij_d_y2(self, ij, tx1, tx2):
        if ij == 11:
            return 0
        elif ij == 12:
            return tx2[1] / np.sqrt(tx2[0]**2 + tx2[1]**2)
        elif ij == 21:
            return 0
        elif ij == 22:
            return (tx2[1] - self.L2) / np.sqrt(tx2[0]**2 + (tx2[1] - self.L2)**2)
        else:
            raise ValueError("Entered tx rx values do not exist")


