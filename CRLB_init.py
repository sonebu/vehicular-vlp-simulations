import math
from functools import lru_cache
import numpy as np
from scipy.integrate import tplquad

class CRLB_init:
    def __init__(self, x1, y1, x2, y2):
        
        self.rxradius = 0.003  # 3mm
        self.c = 3e8  # speed of light(m/s)
        self.fov = 50

        self.m = -np.log(2) / np.log(math.cos(math.radians(self.fov)))
        
        self.L1 = 1
        self.L2 = 1
        
        self.rxxpos, self.rxypos = (0, 0), (0, L2)
        self.rx1 = np.array((self.rxxpos[0], self.rxypos[0]))
        self.rx2 = np.array((self.rxxpos[1], self.rxypos[1]))
        
        self.trxpos, self.trypos = (-5, -5), (2, 3)  # meter
        self.tx1 = np.array((self.trxpos[0], self.trypos[0]))
        self.tx2 = np.array((self.trxpos[1], self.trypos[1]))
        
        self.phi = # koordinatlardan
        self.psi = # koordinatlardan
        self.alpha = # koordinatlardan

    def lamb_coeff(self, ij):
        if ij == 11:
            return ((self.m + 1) * self.rxradius) / (2 * np.pi * (self.tx1[0]**2 + self.tx1[1]**2))
        elif ij == 12:
            return((self.m + 1) * self.rxradius) / (2 * np.pi * (self.tx2[0]**2 + self.tx2[1]**2))
        elif ij == 21:
            return ((self.m + 1) * self.rxradius) / (2 * np.pi * (self.tx1[0]**2 + (self.tx1[1] - self.L2)**2))
        elif ij == 22:
            return ((self.m + 1) * self.rxradius) / (2 * np.pi * (self.tx2[0]**2 + (self.tx2[1] - self.L2)**2))
        else:
            raise ValueError("Entered tx rx values do not exist for coeff")
        
    def lamb_irrad(self, ij): 
        if ij == 11:
            return ((self.tx1[0]/ np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)) * ((self.tx2[1] - self.tx1[1]) / self.L1)
                    - (self.tx1[1] / np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)) * ((self.tx2[0] - self.tx1[0]) / self.L1))
        elif ij == 12:
            return ((self.tx2[0]/ np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)) * ((self.tx2[1] - self.tx1[1]) / self.L1)
                    - (self.tx2[1] / np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)) * ((self.tx2[0] - self.tx1[0]) / self.L1))
        elif ij == 21:
            return ((self.tx1[0]/ np.sqrt(self.tx1[0]**2 + (self.tx1[1] - self.L2)**2))
                      * ((self.tx2[1] - self.tx1[1]) / self.L1) 
                      - ((self.tx1[1] - self.L2) / np.sqrt(self.tx1[0]**2 + (self.tx1[1] - self.L2)**2)) 
                      * ((self.tx2[0] - self.tx1[0]) / self.L1))
        elif ij == 22:
            return ((self.tx2[0]/ np.sqrt(self.tx2[0]**2 + (self.tx2[1] - self.L2)**2)) 
                    * ((self.tx2[1] - self.tx1[1]) / self.L1) 
                    - ((self.tx2[1] - self.L2) / np.sqrt(self.tx2[0]**2 + (self.tx2[1] - self.L2)**2)) 
                    * ((self.tx2[0] - self.tx1[0]) / self.L1))
        else:
            raise ValueError("Entered tx rx values do not exist for irrad angle")

    def lamb_incid(self, ij):
        if ij == 11:
            return self.tx1[0]/ np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)   
        elif ij == 12:
            return self.tx2[0]/ np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)
        elif ij == 21:
            return self.tx1[0]/ np.sqrt(self.tx1[0]**2 + (self.tx1[1] - self.L2)**2)
        elif ij == 22:
            return self.tx2[0]/ np.sqrt(self.tx2[0]**2 + (self.tx2[1] - self.L2)**2)
        else:
            raise ValueError("Entered tx rx values do not exist incidence angle")
    
    def get_h_ij(self, ij, flag):
        
        if(flag):
            return self.lamb_coeff(ij) * (self.lamb_irrad(ij)**(self.m)) * self.lamb_incid(ij)
        else:
            return self.lamb_coeff(ij) * (self.lamb_incid(ij)**(self.m + 1))
    
    
    # d_coeff_x1
    def d_lamb_coeff_x1(self, ij):
        if ij == 11:
            return -(((self.m + 1) * self.rxradius * self.tx1[0]) / np.pi * (self.tx1[0]**2 + self.tx1[1]**2)**2)
        elif ij == 12:
            return 0
        elif ij == 21:
            return -(((self.m + 1) * self.rxradius * self.tx1[0]) / np.pi * (self.tx1[0]**2 + (self.tx1[1] - self.L2)**2)**2)
        elif ij == 22:
            return 0
        else:
            raise ValueError("Entered tx rx values do not exist")
    
    # d_coeff_y1
    def d_lamb_coeff_y1(self, ij):
        if ij == 11:
            return -(((self.m + 1) * self.rxradius * self.tx1[1]) / np.pi * (self.tx1[0]**2 + self.tx1[1]**2)**2)
        elif ij == 12:
            return 0
        elif ij == 21:
            return -(((self.m + 1) * self.rxradius * (self.tx1[1] - self.L2)) 
                 / np.pi * (self.tx1[0]**2 + (self.tx1[1] - self.L2)**2)**2)
        elif ij == 22:
            return 0
        else:
            raise ValueError("Entered tx rx values do not exist")
    
    # d_coeff_x2
    def d_lamb_coeff_x2(self, ij):
        if ij == 11:    
            return 0
        elif ij == 12:
            return -(((self.m + 1) * self.rxradius * self.tx2[0]) / np.pi * (self.tx2[0]**2 + self.tx2[1]**2)**2)
        elif ij == 21:
            return 0
        elif ij == 22:
            return -(((self.m + 1) * self.rxradius * self.tx2[0]) / np.pi * (self.tx2[0]**2 + (self.tx2[1] - self.L2)**2)**2)
        else:
            raise ValueError("Entered tx rx values do not exist")
     
    # d_coeff_y2
    def d_lamb_coeff_y2(self, ij):
        if ij == 11:
            return 0
        elif ij == 12:
            return -(((self.m + 1) * self.rxradius * self.tx2[1]) / np.pi * (self.tx2[0]**2 + self.tx2[1]**2)**2)
        elif ij == 21:
            return 0
        elif ij == 22:
            return -(((self.m + 1) * self.rxradius * (self.tx2[1] - self.L2)) 
                 / np.pi * (self.tx2[0]**2 + (self.tx2[1] - self.L2)**2)**2)
        else:
            raise ValueError("Entered tx rx values do not exist")
   
    # irrad_x1 
    def d_lamb_irrad_x1(self, ij):
        if ij == 11:
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)
            return ((self.tx1[1]**2 / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
                   + (self.tx1[0]*self.tx1[1] / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
                   + self.tx1[1] / (D * L1) 
                   + (self.tx1[0]* (self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
                   - (self.tx1[1]* (self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))
        elif ij == 12:
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)
            return ((self.tx2[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
                   -(self.tx2[1]*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))
        elif ij == 21:   
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx1[0]**2 + (self.tx1[1]-self.L2)**2)
            return (((self.tx1[1]-self.L2)**2 / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
                   + (self.tx1[0]*(self.tx1[1]-self.L2) / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
                   - (self.tx1[1]-self.L2) / (D * L1) 
                   + (self.tx1[0] * (self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
                   - ((self.tx1[1]-self.L2) * (self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))
        elif ij == 22:
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx2[0]**2 + (self.tx2[1]-self.L2)**2)
            return ((self.tx2[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
                   - ((self.tx2[1]-self.L2)*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))
        else:
            raise ValueError("Entered tx rx values do not exist incidence angle")
    
    # irrad_y1
    def d_lamb_irrad_y1(self, ij):
        if ij == 11:
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)
            return (-(self.tx1[0]*self.tx1[1] / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
                   - self.tx1[0] / (D * L1)
                   - (self.tx1[0]**2 / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
                   + (self.tx1[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
                   - (self.tx1[1]*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2)))
        elif ij == 12:
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)
            return ((self.tx2[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
                   - (self.tx2[1]*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2)))
        elif ij == 21:
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx1[0]**2 + (self.tx1[1]-self.L2)**2)
            return (-(self.tx1[0]*(self.tx1[1]-self.L2) / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
                   + self.tx1[0] / (D * L1)
                   - ((self.tx1[0]**2 + self.tx1[1] * (self.tx1[1]-self.L2)) / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
                   + (self.tx1[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
                   - ((self.tx1[1]-self.L2)*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))
        elif ij == 22:
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx2[0]**2 + (self.tx2[1]-self.L2)**2)
            return ((self.tx2[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
                   - ((self.tx2[1]-self.L2)*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2)))
        else:
            raise ValueError("Entered tx rx values do not exist incidence angle")

    # irrad_x2
    def d_lamb_irrad_x2(self, ij):
        if ij == 11:
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)
            return (-(self.tx1[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
                   + (self.tx1[1]*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))
        elif ij == 12:
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)
            return ((self.tx2[1]**2 / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
                   + (self.tx2[0]*self.tx2[1] / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
                   + self.tx2[1] / (D * L1) 
                   - (self.tx2[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
                   + (self.tx2[1]*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))) 
        elif ij == 21:
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx1[0]**2 + (self.tx1[1]-self.L2)**2)
            return (-(self.tx1[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
                   + ((self.tx1[1]-self.L2)*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))
        elif ij == 22:
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx2[0]**2 + (self.tx2[1]-self.L2)**2)
            return (((self.tx2[1]-self.L2)**2 / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
                   + (self.tx2[0]*(self.tx2[1]-self.L2) / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
                   - (self.tx2[1]-self.L2) / (D * L1) 
                   + (self.tx2[0] * (self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
                   - ((self.tx2[1]-self.L2) * (self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))
        else:
            raise ValueError("Entered tx rx values do not exist incidence angle")
            
    # irrad_y2
    def d_lamb_irrad_y2(self, ij):
        if ij == 11:
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)
            return (-(self.tx1[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
                   + (self.tx1[1]*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2)))
        elif ij == 12:
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)
            return (-(self.tx2[0]*self.tx2[1] / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
                   - self.tx2[0] / (D * L1)
                   - (self.tx2[0]**2 / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
                   - (self.tx2[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
                   + (self.tx2[1]*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2)))
        elif ij == 21:
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx1[0]**2 + (self.tx1[1]-self.L2)**2)
            return (-(self.tx1[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
                   + ((self.tx1[1]-self.L2)*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2)))
        elif ij == 22:
            L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
            D = np.sqrt(self.tx2[0]**2 + (self.tx2[1]-self.L2)**2)
            return (-(self.tx2[0]*(self.tx2[1]-self.L2) / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
                   + self.tx2[0] / (D * L1)
                   - ((self.tx2[0]**2 + self.tx2[1] * (self.tx2[1]-self.L2)) / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
                   + (self.tx2[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
                   - ((self.tx2[1]-self.L2)*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2)))
        else: 
            raise ValueError("Entered tx rx values do not exist incidence angle")
    
    # d_incid_x1 
    def d_lamb_incid_x1(self, ij):
        if ij == 11:
            D = np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)
            return self.tx1[1]**2 / (D**3)
        elif ij == 12:
            return 0
        elif ij == 21:
            D = np.sqrt(self.tx1[0]**2 + (self.tx1[1] - self.L2)**2)
            return (self.tx1[1] - self.L2)**2 / (D**3)
        elif ij == 22:
            return 0
        else: 
            raise ValueError("Entered tx rx values do not exist incidence angle")
                    
    # d_incid_y1 
    def d_lamb_incid_y1(self, ij):
        if ij == 11:
            D = np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)
            return -(self.tx1[0]*self.tx1[1]) / (D**3)
        elif ij == 12:
            return 0
        elif ij == 21:
            D = np.sqrt(self.tx1[0]**2 + (self.tx1[1] - self.L2)**2)
            return -(self.tx1[0]*(self.tx1[1] - self.L2)) / (D**3)
        elif ij == 22:
            return 0
        else: 
            raise ValueError("Entered tx rx values do not exist incidence angle")
             
    # d_incid_x2 
    def d_lamb_incid_x2(self, ij):
        if ij == 11:
            return 0
        elif ij == 12:
            D = np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)
            return self.tx2[1]**2 / (D**3)
        elif ij == 21:
            return 0
        elif ij == 22:
            D = np.sqrt(self.tx2[0]**2 + (self.tx2[1] - self.L2)**2)
            return (self.tx2[1] - self.L2)**2 / (D**3)
        else: 
            raise ValueError("Entered tx rx values do not exist incidence angle")
    
    # d_incid_y2 
    def d_lamb_incid_y2(self, ij):
        if ij == 11:
            return 0   
        elif ij == 12:
            D = np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)
            return -(self.tx2[0]*self.tx2[1]) / (D**3) 
        elif ij == 21:
            return 0
        elif ij == 22:
            D = np.sqrt(self.tx2[0]**2 + (self.tx2[1] - self.L2)**2)
            return -(self.tx2[0]*(self.tx2[1] - self.L2)) / (D**3)  
        else: 
            raise ValueError("Entered tx rx values do not exist incidence angle")

            
    # derivatives of hij:  
    def d_h_d_x1(self, ij):      
        return (self.d_lamb_coeff_x1(ij) * self.lamb_irrad(ij) * self.lamb_incid(ij)
                + self.lamb_coeff_x1(ij) * self.d_lamb_irrad_x1(ij) * self.lamb_incid(ij)
                + self.lamb_coeff_x1(ij) * self.lamb_irrad_x1(ij) * self.d_lamb_incid(ij))
    
    def d_h_d_x2(self, ij):
        
        return self.d_lamb_coeff11_x2()
    
    def d_h_d_y1(self, ij):
    
    def d_h_d_y2(self, ij):
    
 