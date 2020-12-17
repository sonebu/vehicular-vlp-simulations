import math
import numpy as np


class CRLB_init:
    """
    This class contains the equations and their derivatives for calculation of CRLB bounds for
    three methods: Robert, Bechadergue, Soner
    """
    def __init__(self, L_1, L_2, rx_area, rx_fov, tx_half_angle, c = 3e8):
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
        self.half_angle = tx_half_angle # angle
        self.m = -np.log(2) / np.log(math.cos(math.radians(self.half_angle)))

    def lamb_coeff(self, ij, tx1, tx2, flag=True):
        """
        (m + 1) * A / (2 * pi * sqrt(tx_x^2 + tx_y^2))
        :param ij:
        :param tx1:
        :param tx2:
        :param flag:
        :return:
        """
        L1 = np.sqrt((tx2[0] - tx1[0]) ** 2 + (tx2[1] - tx1[1]) ** 2)
        if ij == 11:
            return ((self.m + 1) * self.rx_area) / (2 * np.pi * (tx1[0]**2 + tx1[1]**2))
        elif ij == 12:
            if flag:
                return ((self.m + 1) * self.rx_area) / (2 * np.pi * (tx2[0]**2 + tx2[1]**2))
            else:
                return ((self.m + 1) * self.rx_area) / (2 * np.pi * (tx1[0]**2 + (tx1[1] + L1)**2))
        elif ij == 21:
            return ((self.m + 1) * self.rx_area) / (2 * np.pi * (tx1[0]**2 + (tx1[1] - self.L2)**2))
        elif ij == 22:
            if flag:
                return ((self.m + 1) * self.rx_area) / (2 * np.pi * (tx2[0]**2 + (tx2[1] - self.L2)**2))
            else:
                return ((self.m + 1) * self.rx_area) / (2 * np.pi * (tx1[0]**2 + (tx1[1] + self.L1 - self.L2)**2))
        else:
            raise ValueError("Entered tx rx values do not exist for lamb_coeff")
        
    def lamb_irrad(self, ij, tx1, tx2):
        L1 = np.sqrt((tx2[0] - tx1[0]) ** 2 + (tx2[1] - tx1[1]) ** 2)
        if ij == 11:
            return ((tx1[0] / np.sqrt(tx1[0]**2 + tx1[1]**2)) * ((tx2[1] - tx1[1]) / L1)
                    - (tx1[1] / np.sqrt(tx1[0]**2 + tx1[1]**2)) * ((tx2[0] - tx1[0]) / L1))
        elif ij == 12:
            return ((tx2[0] / np.sqrt(tx2[0]**2 + tx2[1]**2)) * ((tx2[1] - tx1[1]) / L1)
                    - (tx2[1] / np.sqrt(tx2[0]**2 + tx2[1]**2)) * ((tx2[0] - tx1[0]) / L1))
        elif ij == 21:
            return ((tx1[0] / np.sqrt(tx1[0]**2 + (tx1[1] - self.L2)**2)) * ((tx2[1] - tx1[1]) / L1)
                    - ((tx1[1] - self.L2) / np.sqrt(tx1[0]**2 + (tx1[1] - self.L2)**2)) * ((tx2[0] - tx1[0]) / L1))
        elif ij == 22:
            return ((tx2[0] / np.sqrt(tx2[0]**2 + (tx2[1] - self.L2)**2)) * ((tx2[1] - tx1[1]) / L1)
                    - ((tx2[1] - self.L2) / np.sqrt(tx2[0]**2 + (tx2[1] - self.L2)**2)) * ((tx2[0] - tx1[0]) / L1))
        else:
            raise ValueError("Entered tx rx values do not exist for irrad angle (lamb_irrad)")

    def lamb_incid(self, ij, tx1, tx2, flag=True):
        L1 = np.sqrt((tx2[0] - tx1[0]) ** 2 + (tx2[1] - tx1[1]) ** 2)
        if ij == 11:
            return tx1[0] / np.sqrt(tx1[0]**2 + tx1[1]**2)
        elif ij == 12:
            if flag:
                return tx2[0] / np.sqrt(tx2[0]**2 + tx2[1]**2)
            else:
                return tx1[0] / np.sqrt(tx1[0]**2 + (tx1[1] + L1)**2)
        elif ij == 21:
            return tx1[0] / np.sqrt(tx1[0]**2 + (tx1[1] - self.L2)**2)
        elif ij == 22:
            if flag:
                return tx2[0] / np.sqrt(tx2[0]**2 + (tx2[1] - self.L2)**2)
            else:
                return tx1[0] / np.sqrt(tx1[0]**2 + (tx1[1] + L1 - self.L2)**2)
        else:
            raise ValueError("lamb_incid")
    
    def get_h_ij(self, ij, tx1, tx2, flag=True):
        if flag:
            return self.lamb_coeff(ij, tx1, tx2, flag) * (self.lamb_irrad(ij, tx1, tx2)**self.m) \
                   * self.lamb_incid(ij, tx1, tx2, flag)
        else:
            return self.lamb_coeff(ij, tx1, tx2, flag) * (self.lamb_incid(ij, tx1, tx2, flag)**(self.m + 1))
    
    
    # d_coeff_x1
    # flag == True for Bechadergue, Soner
    # flag == False for Roberts
    def get_d_lamb_coeff_d_param(self, k, ij, tx1, tx2, flag=True):
        if k == 1:
            return self.d_lamb_coeff_x1(ij, tx1, tx2, flag)
        elif k == 2:
            return self.d_lamb_coeff_y1(ij, tx1, tx2, flag)
        elif k == 3:
            return self.d_lamb_coeff_x2(self, ij, tx1, tx2)
        elif k == 4:
            return self.d_lamb_coeff_y2(self, ij, tx1, tx2)
        else:
            raise ValueError("get_d_lamb_coeff_d_param")
    
    def d_lamb_coeff_x1(self, ij, tx1, tx2, flag=True):
        if ij == 11:
            return -(((self.m + 1) * self.rx_area * tx1[0]) / (np.pi * (tx1[0]**2 + tx1[1]**2)**2))
        elif ij == 12:
            if flag:
                return 0
            else:
                return -(((self.m + 1) * self.rx_area * tx1[0]) / (np.pi * (tx1[0]**2 + (tx1[1] + self.L1)**2)**2))
        elif ij == 21:
            return -(((self.m + 1) * self.rx_area * tx1[0]) / (np.pi * (tx1[0]**2 + (tx1[1] - self.L2)**2)**2))
        elif ij == 22:
            if flag:
                return 0
            else:
                return -(((self.m + 1)*self.rx_area * tx1[0]) / (np.pi * (tx1[0]**2+(tx1[1] + self.L1 - self.L2)**2)**2))
        else:
            raise ValueError("d_lamb_coeff_x1")
    
    # d_coeff_y1
    def d_lamb_coeff_y1(self, ij, tx1, tx2, flag=True):
        if ij == 11:
            return -(((self.m + 1) * self.rx_area * tx1[1]) / (np.pi * (tx1[0]**2 + tx1[1]**2)**2))
        elif ij == 12:
            if flag:
                return 0
            else:
                return -(((self.m + 1) * self.rx_area * (tx1[1] + self.L1)) / (np.pi * (tx1[0]**2 + (tx1[1] + self.L1)**2)**2))
        elif ij == 21:
            return -(((self.m + 1) * self.rx_area * (tx1[1] - self.L2)) / (np.pi * (tx1[0]**2 + (tx1[1] - self.L2)**2)**2))
        elif ij == 22:
            if flag:
                return 0
            else:
                return -(((self.m+1) * self.rx_area * (tx1[1] + self.L1 - self.L2))
                         / (np.pi * (tx1[0]**2 + (tx1[1] + self.L1 - self.L2)**2)**2))
        else:
            raise ValueError("Entered tx rx values do not exist d_lamb_coeff_y1")
    
    # d_coeff_x2
    def d_lamb_coeff_x2(self, ij, tx1, tx2):
        if ij == 11:    
            return 0
        elif ij == 12:
            return -(((self.m + 1) * self.rx_area * tx2[0]) / (np.pi * (tx2[0]**2 + tx2[1]**2)**2))
        elif ij == 21:
            return 0
        elif ij == 22:
            return -(((self.m + 1) * self.rx_area * tx2[0]) / (np.pi * (tx2[0]**2 + (tx2[1] - self.L2)**2)**2))
        else:
            raise ValueError("Entered tx rx values do not exist d_lamb_coeff_x2")
     
    # d_coeff_y2
    def d_lamb_coeff_y2(self, ij, tx1, tx2):
        if ij == 11:
            return 0
        elif ij == 12:
            return -(((self.m + 1) * self.rx_area * tx2[1]) / (np.pi * (tx2[0]**2 + tx2[1]**2)**2))
        elif ij == 21:
            return 0
        elif ij == 22:
            return -(((self.m + 1) * self.rx_area * (tx2[1] - self.L2))
                     / (np.pi * (tx2[0]**2 + (tx2[1] - self.L2)**2)**2))
        else:
            raise ValueError("Entered tx rx values do not exist d_lamb_coeff_y2")

    def get_d_lamb_irrad_d_param(self, k, ij, tx1, tx2):
        if k == 1:
            return self.d_lamb_irrad_x1(ij, tx1, tx2)
        elif k == 2:
            return self.d_lamb_irrad_y1(ij, tx1, tx2)
        elif k == 3:
            return self.d_lamb_irrad_x2(ij, tx1, tx2)
        elif k == 4:
            return self.d_lamb_irrad_y2(ij, tx1, tx2)
        else:
            raise ValueError("Entered tx rx values do not exist")

    # irrad_x1 
    def d_lamb_irrad_x1(self, ij, tx1, tx2):
        L1 = np.sqrt((tx2[0] - tx1[0]) ** 2 + (tx2[1] - tx1[1]) ** 2)
        if ij == 11:
            D = np.sqrt(tx1[0]**2 + tx1[1]**2)
            return ((tx1[1]**2 / D**3) * ((tx2[1] - tx1[1]) / L1)
                   + (tx1[0]*tx1[1] / D**3) * ((tx2[0] - tx1[0]) / L1)
                   + tx1[1] / (D * L1) 
                   + (tx1[0] * (tx2[1] - tx1[1]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2))
                   - (tx1[1] * (tx2[0] - tx1[0]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2)))
        elif ij == 12:
            D = np.sqrt(tx2[0]**2 + tx2[1]**2)
            return ((tx2[0]*(tx2[1] - tx1[1]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2))
                    - (tx2[1]*(tx2[0] - tx1[0]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2))
                    + (tx2[1] / (D * L1)))
        elif ij == 21:
            D = np.sqrt(tx1[0]**2 + (tx1[1]-self.L2)**2)
            return (((tx1[1] - self.L2)**2 / D**3) * ((tx2[1] - tx1[1]) / L1)
                   + (tx1[0] * (tx1[1]-self.L2) / D**3) * ((tx2[0] - tx1[0]) / L1)
                   - (tx1[1] - self.L2) / (D * L1)
                   + (tx1[0] * (tx2[1] - tx1[1]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2))
                   - ((tx1[1] - self.L2) * (tx2[0] - tx1[0]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2)))
        elif ij == 22:
            D = np.sqrt(tx2[0]**2 + (tx2[1]-self.L2)**2)
            return ((tx2[0]*(tx2[1] - tx1[1]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2))
                    - ((tx2[1]-self.L2)*(tx2[0] - tx1[0]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2))
                    + (tx2[1] - self.L2) / (D * L1))
        else:
            raise ValueError("Entered tx rx values do not exist incidence angle")
    
    # irrad_y1
    def d_lamb_irrad_y1(self, ij, tx1, tx2):
        L1 = np.sqrt((tx2[0] - tx1[0]) ** 2 + (tx2[1] - tx1[1]) ** 2)
        if ij == 11:
            D = np.sqrt(tx1[0]**2 + tx1[1]**2)
            return (-(tx1[0]*tx1[1] / D**3) * ((tx2[1] - tx1[1]) / L1)
                   - tx1[0] / (D * L1)
                   - (tx1[0]**2 / D**3) * ((tx2[0] - tx1[0]) / L1)
                   + (tx1[0]*(tx2[1] - tx1[1]) / D) * ((tx2[1] - tx1[1]) / L1) * (L1**(-2))
                   - (tx1[1]*(tx2[0] - tx1[0]) / D) * ((tx2[1] - tx1[1]) / L1) * (L1**(-2)))
        elif ij == 12:
            D = np.sqrt(tx2[0]**2 + tx2[1]**2)
            return ((tx2[0]*(tx2[1] - tx1[1]) / D) * ((tx2[1] - tx1[1]) / L1) * (L1**(-2))
                    - (tx2[1]*(tx2[0] - tx1[0]) / D) * ((tx2[1] - tx1[1]) / L1) * (L1**(-2))
                    - (tx2[0] / (D*L1)))
        elif ij == 21:
            D = np.sqrt(tx1[0]**2 + (tx1[1]-self.L2)**2)
            return (-(tx1[0]*(tx1[1]-self.L2) / D**3) * ((tx2[1] - tx1[1]) / L1)
                   - tx1[0] / (D * L1)  # changed
                   - ((tx1[0]**2 + tx1[1] * (tx1[1]-self.L2)) / D**3) * ((tx2[0] - tx1[0]) / L1)
                   + (tx1[0]*(tx2[1] - tx1[1]) / D) * ((tx2[1] - tx1[1]) / L1) * (L1**(-2))
                   - ((tx1[1]-self.L2)*(tx2[0] - tx1[0]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2)))
        elif ij == 22:
            D = np.sqrt(tx2[0]**2 + (tx2[1]-self.L2)**2)
            return ((tx2[0]*(tx2[1] - tx1[1]) / D) * ((tx2[1] - tx1[1]) / L1) * (L1**(-2))
                    - ((tx2[1]-self.L2)*(tx2[0] - tx1[0]) / D) * ((tx2[1] - tx1[1]) / L1) * (L1**(-2))
                    - (tx2[0] / (D*L1)))
        else:
            raise ValueError("Entered tx rx values do not exist incidence angle")

    # irrad_x2
    def d_lamb_irrad_x2(self, ij, tx1, tx2):
        L1 = np.sqrt((tx2[0] - tx1[0])**2 + (tx2[1] - tx1[1])**2)
        if ij == 11:
            D = np.sqrt(tx1[0]**2 + tx1[1]**2)
            return (-(tx1[0]*(tx2[1] - tx1[1]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2))
                    + (tx1[1]*(tx2[0] - tx1[0]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2))
                    - (tx1[1] / (D * L1)))
        elif ij == 12:
            D = np.sqrt(tx2[0]**2 + tx2[1]**2)
            return ((tx2[1]**2 / D**3) * ((tx2[1] - tx1[1]) / L1)
                   + (tx2[0]*tx2[1] / D**3) * ((tx2[0] - tx1[0]) / L1)
                   - tx2[1] / (D * L1)  # changed
                   - (tx2[0]*(tx2[1] - tx1[1]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2))
                   + (tx2[1]*(tx2[0] - tx1[0]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2))) 
        elif ij == 21:
            D = np.sqrt(tx1[0]**2 + (tx1[1]-self.L2)**2)
            return (-(tx1[0]*(tx2[1] - tx1[1]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2))
                    + ((tx1[1]-self.L2)*(tx2[0] - tx1[0]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2))
                    - ((tx1[1] - self.L2) / (D * L1)))
        elif ij == 22:
            D = np.sqrt(tx2[0]**2 + (tx2[1]-self.L2)**2)
            return (((tx2[1]-self.L2)**2 / D**3) * ((tx2[1] - tx1[1]) / L1)
                   + (tx2[0]*(tx2[1]-self.L2) / D**3) * ((tx2[0] - tx1[0]) / L1)
                   - (tx2[1]-self.L2) / (D * L1) 
                   - (tx2[0] * (tx2[1] - tx1[1]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2))  # changed
                   + ((tx2[1]-self.L2) * (tx2[0] - tx1[0]) / D) * ((tx2[0] - tx1[0]) / L1) * (L1**(-2)))  # changed
        else:
            raise ValueError("Entered tx rx values do not exist incidence angle")
            
    # irrad_y2
    def d_lamb_irrad_y2(self, ij, tx1, tx2):
        L1 = np.sqrt((tx2[0] - tx1[0])**2 + (tx2[1] - tx1[1])**2)
        if ij == 11:
            D = np.sqrt(tx1[0]**2 + tx1[1]**2)
            return (-(tx1[0]*(tx2[1] - tx1[1]) / D) * ((tx2[1] - tx1[1]) / L1) * (L1**(-2))
                    + (tx1[1]*(tx2[0] - tx1[0]) / D) * ((tx2[1] - tx1[1]) / L1) * (L1**(-2))
                    + (tx1[0] / (D * L1)))
        elif ij == 12:
            D = np.sqrt(tx2[0]**2 + tx2[1]**2)
            return (-(tx2[0]*tx2[1] / D**3) * ((tx2[1] - tx1[1]) / L1)
                   + tx2[0] / (D * L1)  # changed
                   - (tx2[0]**2 / D**3) * ((tx2[0] - tx1[0]) / L1)
                   - (tx2[0]*(tx2[1] - tx1[1]) / D) * ((tx2[1] - tx1[1]) / L1) * (L1**(-2))  # changed
                   + (tx2[1]*(tx2[0] - tx1[0]) / D) * ((tx2[1] - tx1[1]) / L1) * (L1**(-2)))  # changed
        elif ij == 21:
            D = np.sqrt(tx1[0]**2 + (tx1[1]-self.L2)**2)
            return (-(tx1[0]*(tx2[1] - tx1[1]) / D) * ((tx2[1] - tx1[1]) / L1) * (L1**(-2))
                    + ((tx1[1]-self.L2)*(tx2[0] - tx1[0]) / D) * ((tx2[1] - tx1[1]) / L1) * (L1**(-2))
                    + (tx1[0] / (D * L1)))
        elif ij == 22:
            D = np.sqrt(tx2[0]**2 + (tx2[1]-self.L2)**2)
            return (-(tx2[0]*(tx2[1]-self.L2) / D**3) * ((tx2[1] - tx1[1]) / L1)
                   + tx2[0] / (D * L1)
                   - ((tx2[0]**2 + tx2[1] * (tx2[1]-self.L2)) / D**3) * ((tx2[0] - tx1[0]) / L1)
                   - (tx2[0]*(tx2[1] - tx1[1]) / D) * ((tx2[1] - tx1[1]) / L1) * (L1**(-2))  # changed
                   + ((tx2[1]-self.L2)*(tx2[0] - tx1[0]) / D) * ((tx2[1] - tx1[1]) / L1) * (L1**(-2)))  # changed
        else: 
            raise ValueError("Entered tx rx values do not exist incidence angle")
    
    # d_incid_x1 
    # flag == True for Bechadergue, Soner
    # flag == False for Roberts
    def get_d_lamb_incid_d_param(self, k, ij, tx1, tx2, flag=True):
        if k == 1:
            return self.d_lamb_incid_x1(ij, tx1, tx2, flag)
        elif k == 2:
            return self.d_lamb_incid_y1(ij, tx1, tx2, flag)
        elif k == 3:
            return self.d_lamb_incid_x2(ij, tx1, tx2, flag)
        elif k == 4:
            return self.d_lamb_incid_y2(ij, tx1, tx2, flag)
        else:
            raise ValueError("Entered tx rx values do not exist incidence angle")

    def d_lamb_incid_x1(self, ij, tx1, tx2, flag=True):
        if ij == 11:
            D = np.sqrt(tx1[0]**2 + tx1[1]**2)
            return tx1[1]**2 / (D**3)
        elif ij == 12:
            if flag:
                return 0
            else:
                D = np.sqrt(tx1[0]**2 + (tx1[1] + self.L1)**2)
                return ((tx1[1] + self.L1)**2) / (D**3)
        elif ij == 21:
            D = np.sqrt(tx1[0]**2 + (tx1[1] - self.L2)**2)
            return (tx1[1] - self.L2)**2 / (D**3)
        elif ij == 22:
            if flag:
                return 0
            else:
                D = np.sqrt(tx1[0]**2 + (tx1[1] + self.L1 - self.L2)**2)
                return ((tx1[1] + self.L1 - self.L2)**2) / (D**3)
        else: 
            raise ValueError("Entered tx rx values do not exist incidence angle")
                    
    # d_incid_y1 
    def d_lamb_incid_y1(self, ij, tx1, tx2, flag=True):
        if ij == 11:
            D = np.sqrt(tx1[0]**2 + tx1[1]**2)
            return -(tx1[0]*tx1[1]) / (D**3)
        elif ij == 12:
            if flag:
                return 0
            else:
                D = np.sqrt(tx1[0]**2 + (tx1[1] + self.L1)**2)
                return -(tx1[0]*(tx1[1] + self.L1)) / (D**3)
        elif ij == 21:
            D = np.sqrt(tx1[0]**2 + (tx1[1] - self.L2)**2)
            return -(tx1[0]*(tx1[1] - self.L2)) / (D**3)
        elif ij == 22:
            if flag:
                return 0
            else:
                D = np.sqrt(tx1[0]**2 + (tx1[1] + self.L1 - self.L2)**2)
                return -(tx1[0]*(tx1[1] + self.L1 - self.L2)) / (D**3)
        else: 
            raise ValueError("Entered tx rx values do not exist incidence angle")
             
    # d_incid_x2 
    def d_lamb_incid_x2(self, ij, tx1, tx2):
        if ij == 11:
            return 0
        elif ij == 12:
            D = np.sqrt(tx2[0]**2 + tx2[1]**2)
            return tx2[1]**2 / (D**3)
        elif ij == 21:
            return 0
        elif ij == 22:
            D = np.sqrt(tx2[0]**2 + (tx2[1] - self.L2)**2)
            return (tx2[1] - self.L2)**2 / (D**3)
        else: 
            raise ValueError("Entered tx rx values do not exist incidence angle")
    
    # d_incid_y2 
    def d_lamb_incid_y2(self, ij, tx1, tx2):
        if ij == 11:
            return 0   
        elif ij == 12:
            D = np.sqrt(tx2[0]**2 + tx2[1]**2)
            return -(tx2[0]*tx2[1]) / (D**3) 
        elif ij == 21:
            return 0
        elif ij == 22:
            D = np.sqrt(tx2[0]**2 + (tx2[1] - self.L2)**2)
            return -(tx2[0]*(tx2[1] - self.L2)) / (D**3)  
        else: 
            raise ValueError("Entered tx rx values do not exist incidence angle")

            
    # derivatives of hij: 
    # flag == True for Bechadergue, Soner
    # flag == False for Roberts
    def get_d_hij_d_param(self, k, ij, tx1, tx2, flag=True):
        if k == 1:
            return self.d_h_d_x1(ij, tx1, tx2, flag)
        elif k == 2:
            return self.d_h_d_y1(ij, tx1, tx2, flag)
        elif k == 3:
            return self.d_h_d_x2(ij, tx1, tx2, flag)
        elif k == 4:
            return self.d_h_d_y2(ij, tx1, tx2, flag)
        else:
            raise ValueError("Entered tx rx values do not exist incidence angle")

    def d_h_d_x1(self, ij, tx1, tx2, flag=True):
        if flag:
            return (self.d_lamb_coeff_x1(ij, tx1, tx2, flag) * (self.lamb_irrad(ij, tx1, tx2)**self.m) * self.lamb_incid(ij, tx1, tx2, flag)
                    + (self.lamb_coeff(ij, tx1, tx2, flag) * (self.m * self.d_lamb_irrad_x1(ij, tx1, tx2) * (self.lamb_irrad(ij, tx1, tx2)**(self.m - 1)))
                       * self.lamb_incid(ij, tx1, tx2, flag))
                    + self.lamb_coeff(ij, tx1, tx2, flag) * (self.lamb_irrad(ij, tx1, tx2)**self.m) * self.d_lamb_incid_x1(ij, tx1, tx2, flag))
        else:
            return (self.d_lamb_coeff_x1(ij, tx1, tx2, flag) * (self.lamb_incid(ij, tx1, tx2, flag)**(self.m + 1))
                    + self.lamb_coeff(ij, tx1, tx2, flag) * (self.m + 1) * self.d_lamb_incid_x1(ij, tx1, tx2, flag) * (self.lamb_incid(ij, tx1, tx2, flag)**self.m))

    def d_h_d_x2(self, ij, tx1, tx2, flag=True):
        
        return (self.d_lamb_coeff_x2(ij, tx1, tx2) * (self.lamb_irrad(ij, tx1, tx2)**self.m) * self.lamb_incid(ij, tx1, tx2, flag)
                + (self.lamb_coeff(ij, tx1, tx2, flag) * (self.m * self.d_lamb_irrad_x2(ij, tx1, tx2) * (self.lamb_irrad(ij, tx1, tx2)**(self.m - 1)))
                   * self.lamb_incid(ij, tx1, tx2))
                + self.lamb_coeff(ij, tx1, tx2, flag) * (self.lamb_irrad(ij, tx1, tx2)**self.m) * self.d_lamb_incid_x2(ij, tx1, tx2))
    
    def d_h_d_y1(self, ij, tx1, tx2, flag=True):
        
        if flag:
            return (self.d_lamb_coeff_y1(ij, tx1, tx2, flag) * (self.lamb_irrad(ij, tx1, tx2)**self.m) * self.lamb_incid(ij, tx1, tx2, flag)
                    + (self.lamb_coeff(ij, tx1, tx2, flag) * (self.m * self.d_lamb_irrad_y1(ij, tx1, tx2) * (self.lamb_irrad(ij, tx1, tx2)**(self.m - 1)))
                       * self.lamb_incid(ij, tx1, tx2, flag))
                    + self.lamb_coeff(ij, tx1, tx2, flag) * (self.lamb_irrad(ij, tx1, tx2)**self.m) * self.d_lamb_incid_y1(ij, tx1, tx2, flag))
        else:
            return (self.d_lamb_coeff_y1(ij, tx1, tx2, flag) * (self.lamb_incid(ij, tx1, tx2)**(self.m + 1))
                    + self.lamb_coeff(ij, tx1, tx2, flag) * (self.m + 1) * self.d_lamb_incid_y1(ij, tx1, tx2, flag) * (self.lamb_incid(ij, tx1, tx2, flag)**self.m))
    
    def d_h_d_y2(self, ij, tx1, tx2, flag=True):
        
        return (self.d_lamb_coeff_y2(ij, tx1, tx2) * (self.lamb_irrad(ij, tx1, tx2)**self.m) * self.lamb_incid(ij, tx1, tx2, flag)
                + (self.lamb_coeff(ij, tx1, tx2, flag) * (self.m * self.d_lamb_irrad_y2(ij, tx1, tx2) * (self.lamb_irrad(ij, tx1, tx2)**(self.m - 1)))
                   * self.lamb_incid(ij, tx1, tx2, flag))
                + self.lamb_coeff(ij, tx1, tx2, flag) * (self.lamb_irrad(ij, tx1, tx2)**self.m) * self.d_lamb_incid_y2(ij, tx1, tx2))
    
    def get_tau(self, ij, tx1, tx2, flag=True):
        
        if ij == 11:    
            return np.sqrt(tx1[0]**2 + tx1[1]**2) / self.c
        elif ij == 12:
            if flag:
                return np.sqrt(tx2[0]**2 + tx2[1]**2) / self.c
            else:
                return np.sqrt(tx1[0]**2 + (tx1[1] + self.L1)**2) / self.c
        elif ij == 21:
            return np.sqrt(tx1[0]**2 + (tx1[1] - self.L2)**2) / self.c
        elif ij == 22:
            if flag:
                return np.sqrt(tx2[0]**2 + (tx2[1] - self.L2)**2) / self.c
            else:
                return np.sqrt(tx1[0]**2 + (tx1[1] + self.L1 - self.L2)**2) / self.c
        else:
            raise ValueError("Entered tx rx values do not exist")
            
    # derivatives of tau:
    def get_d_tau_d_param(self, k, ij, tx1, tx2, flag=True):
        if k == 1:
            return self.d_tau_d_x1(ij, tx1, tx2, flag)
        elif k == 2:
            return self.d_tau_d_y1(ij, tx1, tx2, flag)
        elif k == 3:
            return self.d_tau_d_x2(ij, tx1, tx2, flag)
        elif k == 4:
            return self.d_tau_d_y2(ij, tx1, tx2, flag)
        else:
            raise ValueError("Entered tx rx values do not exist incidence angle")
    
    def d_tau_d_x1(self, ij, tx1, tx2, flag=True):
        
        if ij == 11:    
            return tx1[0] / (np.sqrt(tx1[0]**2 + tx1[1]**2) * self.c)
        elif ij == 12:
            if flag:
                return 0
            else:
                return tx1[0] / (np.sqrt(tx1[0]**2 + (tx1[1] + self.L1)**2) * self.c)
        elif ij == 21:
            return tx1[0] / (np.sqrt(tx1[0]**2 + (tx1[1] - self.L2)**2) * self.c)
        elif ij == 22:
            if flag:
                return 0
            else:
                return tx1[0] / (np.sqrt(tx1[0]**2 + (tx1[1] + self.L1 - self.L2)**2) * self.c)
        else:
            raise ValueError("Entered tx rx values do not exist")
            
    def d_tau_d_x2(self, ij, tx1, tx2, flag=True):
        if flag:
            return 0
        else:
            if ij == 11:
                return 0
            elif ij == 12:
                return tx2[0] / (np.sqrt(tx2[0]**2 + tx2[1]**2) * self.c)
            elif ij == 21:
                return 0
            elif ij == 22:
                return tx2[0] / (np.sqrt(tx2[0]**2 + (tx2[1] - self.L2)**2) * self.c)
            else:
                raise ValueError("Entered tx rx values do not exist")

    def d_tau_d_y1(self, ij, tx1, tx2, flag=True):

        if ij == 11:
            return tx1[1] / (np.sqrt(tx1[0]**2 + tx1[1]**2) * self.c)
        elif ij == 12:
            if flag:
                return 0
            else:
                return (tx1[1] + self.L1) / (np.sqrt(tx1[0]**2 + (tx1[1] + self.L1)**2) * self.c)
        elif ij == 21:
            return (tx1[1] - self.L2) / (np.sqrt(tx1[0]**2 + (tx1[1] - self.L2)**2) * self.c)
        elif ij == 22:
            if flag:
                return 0
            else:
                return (tx1[1]+self.L1-self.L2) / (np.sqrt(tx1[0]**2 + (tx1[1]+self.L1-self.L2)**2) * self.c)
        else:
            raise ValueError("Entered tx rx values do not exist")

    def d_tau_d_y2(self, ij, tx1, tx2, flag=True):
        if flag:
            return 0
        else:
            if ij == 11:
                return 0
            elif ij == 12:
                return tx2[1] / (np.sqrt(tx2[0]**2 + tx2[1]**2) * self.c)
            elif ij == 21:
                return 0
            elif ij == 22:
                return (tx2[1] - self.L2) / (np.sqrt(tx2[0]**2 + (tx2[1] - self.L2)**2) * self.c)
            else:
                raise ValueError("Entered tx rx values do not exist")

    # quad_coeff

    def quad_coeff(self, ij, q, tx1, tx2):
        if q == 1 or q == 3:
            const = -1 / (4*self.fov)
        elif q == 2 or q == 4:
            const = 1 / (4*self.fov)
        else:
            raise ValueError("Entered q value does not exist")

        if ij == 11:
            return (1/4) + const*np.arctan(tx1[1] / tx1[0])
        elif ij == 12:
            return (1/4) + const*np.arctan(tx2[1] / tx2[0])
        elif ij == 21:
            return (1/4) + const*np.arctan((tx1[1] - self.L2) / tx1[0])
        elif ij == 22:
            return (1/4) + const*np.arctan((tx2[1] - self.L2) / tx2[0])
        else:
            raise ValueError("Entered tx rx values do not exist")

    # derivatives of quad_coeff
    # buradan itibaren indentlere bir bakilsin
    def d_quad_coeff_d_param(self, k, ij, q, tx1, tx2):
        if k == 1:
            return self.d_quad_coeff_d_x1(ij, q, tx1, tx2)
        elif k == 2:
            return self.d_quad_coeff_d_y1(ij, q, tx1, tx2)
        elif k == 3:
            return self.d_quad_coeff_d_x2(ij, q, tx1, tx2)
        elif k == 4:
            return self.d_quad_coeff_d_y2(ij, q, tx1, tx2)
        else:
            raise ValueError("Entered tx rx values do not exist incidence angle")

    def d_quad_coeff_d_x1(self, ij, q, tx1, tx2):
        if q == 1 or q == 3:
            const = -1 / (4*self.fov)
        elif q == 2 or q == 4:
            const = 1 / (4*self.fov)
        else:
            raise ValueError("Entered q value does not exist")

        if ij == 11:
            return -const*(tx1[1] / (tx1[0]**2 + tx1[1]**2))
        elif ij == 12:
            return 0
        elif ij == 21:
            return -const*((tx1[1] - self.L2) / (tx1[0]**2 + (tx1[1] - self.L2)**2))
        elif ij == 22:
            return 0
        else:
            raise ValueError("Entered tx rx values do not exist")

    def d_quad_coeff_d_x2(self, ij, q, tx1, tx2):

        if q == 1 or q == 3:
            const = -1 / (4*self.fov)
        elif q == 2 or q == 4:
            const = 1 / (4*self.fov)
        else:
            raise ValueError("Entered q value does not exist")

        if ij == 11:
            return 0
        elif ij == 12:
            return -const*(tx2[1] / (tx1[0]**2 + tx1[1]**2))
        elif ij == 21:
            return 0
        elif ij == 22:
            return -const*((tx2[1] - self.L2) / (tx2[0]**2 + (tx2[1] - self.L2)**2))
        else:
            raise ValueError("Entered tx rx values do not exist")

    def d_quad_coeff_d_y1(self, ij, q, tx1, tx2):

        if q == 1 or q == 3:
            const = -1 / (4*self.fov)
        elif q == 2 or q == 4:
            const = 1 / (4*self.fov)
        else:
            raise ValueError("Entered q value does not exist")

        if ij == 11:
            return const*(tx1[0] / (tx1[0]**2 + tx1[1]**2))
        elif ij == 12:
            return 0
        elif ij == 21:
            return const*(tx1[0] / (tx1[0]**2 + (tx1[1] - self.L2)**2))
        elif ij == 22:
            return 0
        else:
            raise ValueError("Entered tx rx values do not exist")

    def d_quad_coeff_d_y2(self, ij, q, tx1, tx2):

        if q == 1 or q == 3:
            const = -1 / (4*self.fov)
        elif q == 2 or q == 4:
            const = 1 / (4*self.fov)
        else:
            raise ValueError("Entered q value does not exist")

        if ij == 11:
            return 0
        elif ij == 12:
            return const*(tx2[0] / (tx1[0]**2 + tx1[1]**2))
        elif ij == 21:
            return 0
        elif ij == 22:
            return const*(tx2[0] / (tx2[0]**2 + (tx2[1] - self.L2)**2))
        else:
            raise ValueError("Entered tx rx values do not exist")

    # h_ijq
    def get_h_ijq(self, ij, q, tx1, tx2):
        return self.get_h_ij(ij, tx1, tx2) * self.quad_coeff(ij, q, tx1, tx2)

    # derivatives of h_ijq
    def get_d_hij_q_d_param(self, k, ij, q, tx1, tx2):
        if k == 1:
            return self.d_hij_q_d_x1(ij, q, tx1, tx2)
        elif k == 2:
            return self.d_hij_q_d_y1(ij, q, tx1, tx2)
        elif k == 3:
            return self.d_hij_q_d_x2(ij, q, tx1, tx2)
        elif k == 4:
            return self.d_hij_q_d_y2(ij, q, tx1, tx2)
        else:
            raise ValueError("Entered tx rx values do not exist")

    def d_hij_q_d_x1(self, ij, q, tx1, tx2):
        return self.d_h_d_x1(ij, tx1, tx2, True) * self.quad_coeff(ij,q, tx1, tx2) + self.get_h_ij(ij, tx1, tx2) * self.d_quad_coeff_d_x1(ij, q, tx1, tx2)

    def d_hij_q_d_x2(self, ij, q, tx1, tx2):
        return self.d_h_d_x2(ij, tx1, tx2, True) * self.quad_coeff(ij,q, tx1, tx2) + self.get_h_ij(ij, tx1, tx2) * self.d_quad_coeff_d_x2(ij, q, tx1, tx2)

    def d_hij_q_d_y1(self, ij, q, tx1, tx2):
        return self.d_h_d_y1(ij, tx1, tx2, True) * self.quad_coeff(ij,q, tx1, tx2) + self.get_h_ij(ij, tx1, tx2) * self.d_quad_coeff_d_y1(ij, q, tx1, tx2)

    def d_hij_q_d_y2(self, ij, q, tx1, tx2):
        return self.d_h_d_y2(ij, tx1, tx2, True) * self.quad_coeff(ij,q, tx1, tx2) + self.get_h_ij(ij, tx1, tx2) * self.d_quad_coeff_d_y2(ij, q, tx1, tx2)