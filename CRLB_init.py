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
            raise ValueError("Entered tx rx values do not exist")
        
    def lamb_irrad11(self): 
        
        return ((self.tx1[0]/ np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)) * ((self.tx2[1] - self.tx1[1]) / self.L1)
                - (self.tx1[1] / np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)) * ((self.tx2[0] - self.tx1[0]) / self.L1))
        
    def lamb_incid11(self):
        
        return self.tx1[0]/ np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)    
    
    def get_h11(self, flag):
        
        if(flag):
            return self.lamb_coeff11() * (self.lamb_irrad11()**(self.m)) * self.lamb_incid11()
        else:
            return self.lamb_coeff11() * (self.lamb_incid11()**(self.m + 1))
        
    def lamb_irrad12(self): 
        
        return ((self.tx2[0]/ np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)) * ((self.tx2[1] - self.tx1[1]) / self.L1)
                - (self.tx2[1] / np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)) * ((self.tx2[0] - self.tx1[0]) / self.L1))
        
    def lamb_incid12(self):
        
        return self.tx2[0]/ np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)
        
    
    def get_h12(self, flag):
        
        if(flag):
            return self.lamb_coeff12() * (self.lamb_irrad12()**(self.m)) * self.lamb_incid12()
        else:
            return self.lamb_coeff12() * (self.lamb_incid12()**(self.m + 1))
        
    def lamb_irrad21(self):
        
        return ((self.tx1[0]/ np.sqrt(self.tx1[0]**2 + (self.tx1[1] - self.L2)**2))
                      * ((self.tx2[1] - self.tx1[1]) / self.L1) 
                      - ((self.tx1[1] - self.L2) / np.sqrt(self.tx1[0]**2 + (self.tx1[1] - self.L2)**2)) 
                      * ((self.tx2[0] - self.tx1[0]) / self.L1))
        
    def lamb_incid21(self):
        
        return self.tx1[0]/ np.sqrt(self.tx1[0]**2 + (self.tx1[1] - self.L2)**2)
        
    def get_h21(self, flag):
        
        if(flag):
            return self.lamb_coeff21() * (self.lamb_irrad21()**(self.m)) * self.lamb_incid21()
        else:
            return self.lamb_coeff21() * (self.lamb_incid21()**(self.m + 1))

    def lamb_irrad22(self):
        
        return ((self.tx2[0]/ np.sqrt(self.tx2[0]**2 + (self.tx2[1] - self.L2)**2)) 
                * ((self.tx2[1] - self.tx1[1]) / self.L1) 
                - ((self.tx2[1] - self.L2) / np.sqrt(self.tx2[0]**2 + (self.tx2[1] - self.L2)**2)) 
                * ((self.tx2[0] - self.tx1[0]) / self.L1))
        
    def lamb_incid22(self):
        
        return self.tx2[0]/ np.sqrt(self.tx2[0]**2 + (self.tx2[1] - self.L2)**2)
    
    def get_h22(self, flag):
        
        if(flag):
            return self.lamb_coeff22() * (self.lamb_irrad22()**(self.m)) * self.lamb_incid22()
        else:
            return self.lamb_coeff22() * (self.lamb_incid22()**(self.m + 1))
    
    # coeff11
    
    def d_lamb_coeff11_x1(self):
  
        return -(((self.m + 1) * self.rxradius * self.tx1[0]) / np.pi * (self.tx1[0]**2 + self.tx1[1]**2)**2)

    def d_lamb_coeff11_y1(self):
  
        return -(((self.m + 1) * self.rxradius * self.tx1[1]) / np.pi * (self.tx1[0]**2 + self.tx1[1]**2)**2)

    def d_lamb_coeff11_x2(self):
  
        return 0

    def d_lamb_coeff11_y2(self):
  
        return 0

    # coeff12

    def d_lamb_coeff12_x1(self):
  
        return 0

    def d_lamb_coeff12_y1(self):
  
        return 0

    def d_lamb_coeff12_x2(self):
  
        return -(((self.m + 1) * self.rxradius * self.tx2[0]) / np.pi * (self.tx2[0]**2 + self.tx2[1]**2)**2)

    def d_lamb_coeff12_y2(self):
  
        return -(((self.m + 1) * self.rxradius * self.tx2[1]) / np.pi * (self.tx2[0]**2 + self.tx2[1]**2)**2)

    # coeff21
    
    def d_lamb_coeff21_x1(self):
  
        return -(((self.m + 1) * self.rxradius * self.tx1[0]) / np.pi * (self.tx1[0]**2 + (self.tx1[1] - self.L2)**2)**2)

    def d_lamb_coeff21_y1(self):
  
        return -(((self.m + 1) * self.rxradius * (self.tx1[1] - self.L2)) 
                 / np.pi * (self.tx1[0]**2 + (self.tx1[1] - self.L2)**2)**2)

    def d_lamb_coeff21_x2(self):
  
        return 0

    def d_lamb_coeff21_y2(self):
  
        return 0

    # coeff22
    
    def d_lamb_coeff22_x1(self):
  
        return 0

    def d_lamb_coeff22_y1(self):
  
        return 0

    def d_lamb_coeff22_x2(self):
  
        return -(((self.m + 1) * self.rxradius * self.tx2[0]) / np.pi * (self.tx2[0]**2 + (self.tx2[1] - self.L2)**2)**2)

    def d_lamb_coeff22_y2(self):
  
        return -(((self.m + 1) * self.rxradius * (self.tx2[1] - self.L2)) 
                 / np.pi * (self.tx2[0]**2 + (self.tx2[1] - self.L2)**2)**2)

    # irrad11
    
    def d_lamb_irrad11_x1(self):
        
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)
        return ((self.tx1[1]**2 / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
               + (self.tx1[0]*self.tx1[1] / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
               + self.tx1[1] / (D * L1) 
               + (self.tx1[0]* (self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
               - (self.tx1[1]* (self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))

    def d_lamb_irrad11_y1(self):
        
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)
        return (-(self.tx1[0]*self.tx1[1] / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
               - self.tx1[0] / (D * L1)
               - (self.tx1[0]**2 / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
               + (self.tx1[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
               - (self.tx1[1]*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2)))

    def d_lamb_irrad11_x2(self):
  
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)
        return (-(self.tx1[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
               + (self.tx1[1]*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))

    def d_lamb_irrad11_y2(self):
        
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)
        return (-(self.tx1[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
               + (self.tx1[1]*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2)))

    # irrad12
    
    def d_lamb_irrad12_x1(self):
        
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)
        return ((self.tx2[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
               -(self.tx2[1]*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))

    def d_lamb_irrad12_y1(self):
        
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)
        return ((self.tx2[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
               - (self.tx2[1]*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2)))
        
    def d_lamb_irrad12_x2(self):
  
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)
        return ((self.tx2[1]**2 / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
               + (self.tx2[0]*self.tx2[1] / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
               + self.tx2[1] / (D * L1) 
               - (self.tx2[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
               + (self.tx2[1]*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))) 


    def d_lamb_irrad12_y2(self):
        
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)
        return (-(self.tx2[0]*self.tx2[1] / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
               - self.tx2[0] / (D * L1)
               - (self.tx2[0]**2 / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
               - (self.tx2[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
               + (self.tx2[1]*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2)))
    
    # irrad21
    
    def d_lamb_irrad21_x1(self):
        
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx1[0]**2 + (self.tx1[1]-self.L2)**2)
        return (((self.tx1[1]-self.L2)**2 / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
               + (self.tx1[0]*(self.tx1[1]-self.L2) / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
               - (self.tx1[1]-self.L2) / (D * L1) 
               + (self.tx1[0] * (self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
               - ((self.tx1[1]-self.L2) * (self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))

    def d_lamb_irrad21_y1(self):
        
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx1[0]**2 + (self.tx1[1]-self.L2)**2)
        return (-(self.tx1[0]*(self.tx1[1]-self.L2) / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
               + self.tx1[0] / (D * L1)
               - ((self.tx1[0]**2 + self.tx1[1] * (self.tx1[1]-self.L2)) / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
               + (self.tx1[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
               - ((self.tx1[1]-self.L2)*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))

    def d_lamb_irrad21_x2(self):
  
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx1[0]**2 + (self.tx1[1]-self.L2)**2)
        return (-(self.tx1[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
               + ((self.tx1[1]-self.L2)*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))

    def d_lamb_irrad21_y2(self):
        
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx1[0]**2 + (self.tx1[1]-self.L2)**2)
        return (-(self.tx1[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
               + ((self.tx1[1]-self.L2)*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2)))
    
    # irrad22
    
    def d_lamb_irrad22_x1(self):
        
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx2[0]**2 + (self.tx2[1]-self.L2)**2)
        return ((self.tx2[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
               - ((self.tx2[1]-self.L2)*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))

    def d_lamb_irrad22_y1(self):
    
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx2[0]**2 + (self.tx2[1]-self.L2)**2)
        return ((self.tx2[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
               - ((self.tx2[1]-self.L2)*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2)))

    def d_lamb_irrad22_x2(self):
  
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx2[0]**2 + (self.tx2[1]-self.L2)**2)
        return (((self.tx2[1]-self.L2)**2 / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
               + (self.tx2[0]*(self.tx2[1]-self.L2) / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
               - (self.tx2[1]-self.L2) / (D * L1) 
               + (self.tx2[0] * (self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2))
               - ((self.tx2[1]-self.L2) * (self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[0] - self.tx1[0]) / L1) * (L1**(-2)))

    def d_lamb_irrad22_y2(self):
        
        L1 = np.sqrt((self.tx2[0] - self.tx1[0])**2 + (self.tx2[1] - self.tx1[1])**2)
        D = np.sqrt(self.tx2[0]**2 + (self.tx2[1]-self.L2)**2)
        return (-(self.tx2[0]*(self.tx2[1]-self.L2) / D**3) * ((self.tx2[1] - self.tx1[1]) / L1)
               + self.tx2[0] / (D * L1)
               - ((self.tx2[0]**2 + self.tx2[1] * (self.tx2[1]-self.L2)) / D**3) * ((self.tx2[0] - self.tx1[0]) / L1)
               + (self.tx2[0]*(self.tx2[1] - self.tx1[1]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2))
               - ((self.tx2[1]-self.L2)*(self.tx2[0] - self.tx1[0]) / D) * ((self.tx2[1] - self.tx1[1]) / L1) * (L1**(-2)))
    
    # incid11
        
    def d_lamb_incid11_x1(self):
        
        D = np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)
        return self.tx1[1]**2 / (D**3)

    def d_lamb_incid11_y1(self):
        D = np.sqrt(self.tx1[0]**2 + self.tx1[1]**2)
        return -(self.tx1[0]*self.tx1[1]) / (D**3)

    def d_lamb_incid11_x2(self):
        return 0

    def d_lamb_incid11_y2(self):
        return 0    

    # incid12
        
    def d_lamb_incid12_x1(self):
        return 0

    def d_lamb_incid12_y1(self):
        return 0

    def d_lamb_incid12_x2(self):
        D = np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)
        return self.tx2[1]**2 / (D**3)

    def d_lamb_incid12_y2(self):
        D = np.sqrt(self.tx2[0]**2 + self.tx2[1]**2)
        return -(self.tx2[0]*self.tx2[1]) / (D**3) 
    
    # incid21
    
    def d_lamb_incid21_x1(self):
        D = np.sqrt(self.tx1[0]**2 + (self.tx1[1] - self.L2)**2)
        return (self.tx1[1] - self.L2)**2 / (D**3)

    def d_lamb_incid21_y1(self):
        D = np.sqrt(self.tx1[0]**2 + (self.tx1[1] - self.L2)**2)
        return -(self.tx1[0]*(self.tx1[1] - self.L2)) / (D**3)

    def d_lamb_incid21_x2(self):
        return 0

    def d_lamb_incid21_y2(self):
        return 0    

    # incid22
        
    def d_lamb_incid22_x1(self):
        return 0

    def d_lamb_incid22_y1(self):
        return 0

    def d_lamb_coeff22_x2(self):
        D = np.sqrt(self.tx2[0]**2 + (self.tx2[1] - self.L2)**2)
        return (self.tx2[1] - self.L2)**2 / (D**3)

    def d_lamb_coeff12_y2(self):
        D = np.sqrt(self.tx2[0]**2 + (self.tx2[1] - self.L2)**2)
        return -(self.tx2[0]*(self.tx2[1] - self.L2)) / (D**3)  
    
    # derivatives of hij:
    
    def d_h11_d_x1(self):
        
        return (self.d_lamb_coeff11_x1() * self.lamb.irrad11() * self.lamb.incid11()
                + self.lamb_coeff11_x1() * self.d_lamb.irrad11_x1() * self.lamb.incid11()
                + self.lamb_coeff11_x1() * self.lamb.irrad11_x1() * self.d_lamb.incid11())
    
    def d_h11_d_x2(self):
        
        return self.d_lamb_coeff11_x2()
    
    def d_h11_d_y1(self):
    
    def d_h11_d_y2(self):
    
    def d_h12_d_x1(self):
    
    def d_h12_d_x2(self):
    
    def d_h12_d_y1(self):
    
    def d_h12_d_y2(self):
    
    def d_h21_d_x1(self):
    
    def d_h21_d_x2(self):
    
    def d_h21_d_y1(self):
    
    def d_h21_d_y2(self):
    
    def d_h22_d_x1(self):
    
    def d_h22_d_x2(self):
    
    def d_h22_d_y1(self):
    
    def d_h22_d_y2(self):