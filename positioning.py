import numpy as np

################################################################################################################
### CRLBs
################################################################################################################

def crlb_classicalfix(w1, w2, dg1_dp1, dg1_dp2, dg2_dp1, dg2_dp2):
    fim = np.zeros((2,2))
    fim[0,0] = -( (1/(w1**2))*(dg1_dp1*dg1_dp1) + (1/(w2**2))*(dg2_dp1*dg2_dp1) ) 
    fim[0,1] = -( (1/(w1**2))*(dg1_dp1*dg1_dp2) + (1/(w2**2))*(dg2_dp1*dg2_dp2) ) 
    fim[1,0] = fim[0,1]; # symmetric
    fim[1,1] = -( (1/(w1**2))*(dg1_dp2*dg1_dp2) + (1/(w2**2))*(dg2_dp2*dg2_dp2) ) 

    crlb = -np.linalg.inv(fim);

    var_x = crlb[0,0]
    var_y = crlb[1,1]

    return var_x, var_y

def classicalfix_directbearing_crlb(W_aoaL, W_aoaR, x, y, L):
    d_aoaL_dx =  y/(x**2+y**2)
    d_aoaR_dx =  y/((x-L)**2+y**2)
    d_aoaL_dy = -x/(x**2+y**2)
    d_aoaR_dy = -(x-L)/((x-L)**2+y**2)

    var_x, var_y = crlb_classicalfix(W_aoaL, W_aoaR, d_aoaL_dx, d_aoaL_dy, d_aoaR_dx, d_aoaR_dy);

    return var_x, var_y

def classicalfix_directrange_crlb(W_dL, W_dR, x, y, L):
    d_dL_dx = x/(np.sqrt(x**2+y**2))
    d_dR_dx = (x-L)/(np.sqrt((x-L)**2+y**2))
    d_dL_dy = y/(np.sqrt(x**2+y**2))
    d_dR_dy = y/(np.sqrt((x-L)**2+y**2))

    var_x, var_y = crlb_classicalfix(W_dL, W_dR, d_dL_dx, d_dL_dy, d_dR_dx, d_dR_dy);
        
    return var_x, var_y

def classicalfix_diffbearing_crlb(W_delaoaLR_tx1, W_delaoaLR_tx2, x, y, L):
    d_delaoaLR_tx1_dx = y/(x**2+y**2) - y/((x-L)**2+y**2)
    d_delaoaLR_tx2_dx = y/((x+L)**2+y**2) - y/(x**2+y**2)
    d_delaoaLR_tx1_dy = (x-L)/((x-L)**2+y**2) - x/(x**2+y**2)
    d_delaoaLR_tx2_dy = x/(x**2+y**2) - (x+L)/((x+L)**2+y**2)

    var_x, var_y = crlb_classicalfix(W_delaoaLR_tx1, W_delaoaLR_tx2, d_delaoaLR_tx1_dx, d_delaoaLR_tx1_dy, d_delaoaLR_tx2_dx, d_delaoaLR_tx2_dy);
        
    return var_x, var_y

def classicalfix_diffrange_crlb(W_deldLR_tx1, W_deldLR_tx2, x, y, L):
    d_deldLR_tx1_dx = x/(np.sqrt(x**2+y**2)) - (x-L)/(np.sqrt((x-L)**2+y**2))
    d_deldLR_tx2_dx = (x+L)/(np.sqrt((x+L)**2+y**2)) - x/(np.sqrt(x**2+y**2))
    d_deldLR_tx1_dy = y/(np.sqrt(x**2+y**2)) - y/(np.sqrt((x-L)**2+y**2))
    d_deldLR_tx2_dy = y/(np.sqrt((x+L)**2+y**2)) - y/(np.sqrt(x**2+y**2))

    var_x, var_y = crlb_classicalfix(W_deldLR_tx1, W_deldLR_tx2, d_deldLR_tx1_dx, d_deldLR_tx1_dy, d_deldLR_tx2_dx, d_deldLR_tx2_dy);
        
    return var_x, var_y

def crlb_runningfix(w1, w2, dg1_dp1, dg1_dp2, dg2_dp1, dg2_dp2):
    fim = np.zeros((2,2))
    fim[0,0] = -( (1/(w1**2))*(dg1_dp1*dg1_dp1) + (1/(w2**2))*(dg2_dp1*dg2_dp1) ) 
    fim[0,1] = -( (1/(w1**2))*(dg1_dp1*dg1_dp2) + (1/(w2**2))*(dg2_dp1*dg2_dp2) ) 
    fim[1,0] = fim[0,1]; # symmetric
    fim[1,1] = -( (1/(w1**2))*(dg1_dp2*dg1_dp2) + (1/(w2**2))*(dg2_dp2*dg2_dp2) ) 

    crlb = -np.linalg.inv(fim);

    var_x = crlb[0,0]
    var_y = crlb[1,1]

    return var_x, var_y

### left for later
### def runningfix_directbearing_crlb(W_aoaL, W_aoaR, x, y, L):
###     d_aoaL_dx =  y/(x**2+y**2)
###     d_aoaR_dx =  y/((x-L)**2+y**2)
###     d_aoaL_dy = -x/(x**2+y**2)
###     d_aoaR_dy = -(x-L)/((x-L)**2+y**2)
### 
###     var_x, var_y = crlb_classicalfix(W_aoaL, W_aoaR, d_aoaL_dx, d_aoaL_dy, d_aoaR_dx, d_aoaR_dy);
### 
###     return var_x, var_y
### 
### def runningfix_directrange_crlb(W_dL, W_dR, x, y, L):
###     d_dL_dx = x/(np.sqrt(x**2+y**2))
###     d_dR_dx = (x-L)/(np.sqrt((x-L)**2+y**2))
###     d_dL_dy = y/(np.sqrt(x**2+y**2))
###     d_dR_dy = y/(np.sqrt((x-L)**2+y**2))
### 
###     var_x, var_y = crlb_classicalfix(W_dL, W_dR, d_dL_dx, d_dL_dy, d_dR_dx, d_dR_dy);
###         
###     return var_x, var_y


################################################################################################################
### Algorithms
################################################################################################################

def classicalfix_directbearing_mle(aoaL, aoaR, L):
	est_x = L*(1 + np.sin(np.deg2rad(aoaR))*np.cos(np.deg2rad(aoaL))/np.sin(np.deg2rad(aoaL - aoaR)));
	est_y = L*np.cos(np.deg2rad(aoaR))*np.cos(np.deg2rad(aoaL))/np.sin(np.deg2rad(aoaL - aoaR));
	return est_x, est_y 

def classicalfix_directrange_mle(dL, dR, L):
    est_x = (dL**2 - dR**2 + L**2)/(2*L)
    est_y = np.sqrt(dL**2-est_x**2)
    return est_x, est_y

def classicalfix_diffbearing_mle(delaoaLR_tx1, delaoaLR_tx2, L):
    K = 0.5*(1/np.tan(np.deg2rad(delaoaLR_tx2))-1/np.tan(np.deg2rad(delaoaLR_tx1)));
    est_y = (1/(1+K**2))*(L/np.tan(np.deg2rad(delaoaLR_tx2))-L*K);
    est_x = K*est_y;
    return est_x, est_y

def classicalfix_diffrange_mle(deldLR_tx1, deldLR_tx2, L):
    #following notations such as Y_A, D, A, and B are in line with the paper itself
    Y_A = L
    D   = L

    #calculate x,y position of the leading vehicle using eqs. in Robert's method
    if ((np.abs(deldLR_tx1) > 1e-4) and (np.abs(deldLR_tx2) > 1e-4)):
        A = Y_A ** 2 * (1 / (deldLR_tx1 ** 2) - 1 / (deldLR_tx2 ** 2))

        B1 = (-(Y_A ** 3) + 2 * (Y_A ** 2) * D + Y_A * (deldLR_tx1 ** 2)) / (deldLR_tx1 ** 2)
        B2 = (-(Y_A ** 3) + Y_A * (deldLR_tx2 ** 2)) / (deldLR_tx2 ** 2)
        B = B1 - 2 * D - B2

        C1 = ((Y_A ** 4) + 4 * (D ** 2) * (Y_A ** 2) + (deldLR_tx1 ** 4) - 4 * D * (Y_A ** 3) - 2 * (Y_A ** 2) * (deldLR_tx1 ** 2) + 4 * D * Y_A * (deldLR_tx1 ** 2)) / (4 * (deldLR_tx1 ** 2))
        C2 = ((Y_A ** 4) + (deldLR_tx2 ** 4) - 2 * (Y_A ** 2) * (deldLR_tx2 ** 2)) / (4 * (deldLR_tx2 ** 2))
        C = C1 - D ** 2 - C2

        if deldLR_tx1 * deldLR_tx2 > 0:
            Y_B = (- B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
        else:
            Y_B = (- B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
        if ( ((Y_A ** 2 - 2 * Y_A * Y_B - deldLR_tx2 ** 2) / (2 * deldLR_tx2)) ** 2 - (Y_B ** 2) < 0 ):
            # since assumes parallel, fails to find close delays -> negative in srqt
            out = (np.nan, np.nan, np.nan, np.nan)
        X_A = - np.sqrt(((Y_A ** 2 - 2 * Y_A * Y_B - deldLR_tx2 ** 2) / (2 * deldLR_tx2)) ** 2 - (Y_B ** 2))
    elif (np.abs(deldLR_tx1) <= 1e-4):
        Y_B = Y_A / 2 - D
        if ((2 * D * Y_A - deldLR_tx2 ** 2) / (2 - deldLR_tx2)) ** 2 - (D - Y_A / 2) ** 2 < 0:
            # since assumes parallel, fails to find close delays -> negative in srqt
            out = (np.nan, np.nan, np.nan, np.nan)
        X_A = - np.sqrt(((2 * D * Y_A - deldLR_tx2 ** 2) / (2 - deldLR_tx2)) ** 2 - (D - Y_A / 2) ** 2)
    else:
        Y_B = Y_A / 2
        if ((2 * Y_A * D + deldLR_tx1 ** 2) / (2 * deldLR_tx1)) ** 2 - (D + Y_A / 2) ** 2 < 0:
            # since assumes parallel, fails to find close delays -> negative in srqt
            out = (np.nan, np.nan, np.nan, np.nan)
        X_A = - np.sqrt(((2 * Y_A * D + deldLR_tx1 ** 2) / (2 * deldLR_tx1)) ** 2 - (D + Y_A / 2) ** 2)

    # denklemler elec491 convention'ına göre yazıldığı için bir sign ve transpose operasyonu var
    est_x = (0-Y_B)
    est_y = -X_A
    #x_txR = (0-Y_B) + L
    #y_txR = -X_A # see the parallel assumption here

    return est_x, est_y

def runningfix_directbearing_mle(aoa_t0, aoa_t1, dv, av):
    gamma = np.tan(np.deg2rad(aoa_t0))
    beta = np.tan(np.deg2rad(aoa_t1))
    est_y_t0 = dv*( ( np.cos(av) - beta*np.sin(av) )/( beta - gamma ) )
    est_x_t0 = gamma * est_y_t0
    est_y_t1 = np.sin(av)*dv + est_y_t0
    est_x_t1 = beta * est_y_t1
    return est_x_t0, est_y_t0, est_x_t1, est_y_t1

def runningfix_directrange_mle(d_t0, d_t1, dv, av):
    argument = (-d_t0**2 + dv**2 + d_t1**2)/(2*d_t1*dv)
    argument = np.maximum(-1, np.minimum(1,argument))
    varphi = av + np.arccos( argument )
    est_y_t1 = d_t1*np.sin(varphi)
    est_x_t1 = d_t1*np.cos(varphi)
    est_y_t0 = est_y_t1 - dv*np.sin(av)
    est_x_t0 = est_x_t1 - dv*np.cos(av)
    return est_x_t0, est_y_t0, est_x_t1, est_y_t1



#############################################
### 2*rtof, L -> xu
