import numpy as np
#############################################
### 2*aoa, L  -> sonercoleri tvt
def aoa2_positioning(aoa_rxL_txL, aoa_rxL_txR, aoa_rxR_txL, aoa_rxR_txR, L):
	est_x_txL = L*(1 + np.sin(np.deg2rad(aoa_rxR_txL))*np.cos(np.deg2rad(aoa_rxL_txL))/np.sin(np.deg2rad(aoa_rxL_txL - aoa_rxR_txL)));
	est_y_txL = L*np.cos(np.deg2rad(aoa_rxR_txL))*np.cos(np.deg2rad(aoa_rxL_txL))/np.sin(np.deg2rad(aoa_rxL_txL - aoa_rxR_txL));
	est_x_txR = L*(1 + np.sin(np.deg2rad(aoa_rxR_txR))*np.cos(np.deg2rad(aoa_rxL_txR))/np.sin(np.deg2rad(aoa_rxL_txR - aoa_rxR_txR)));
	est_y_txR = L*np.cos(np.deg2rad(aoa_rxR_txR))*np.cos(np.deg2rad(aoa_rxL_txR))/np.sin(np.deg2rad(aoa_rxL_txR - aoa_rxR_txR));

	return est_x_txL, est_y_txL, est_x_txR, est_y_txR 

#############################################
### aoa_t, aoa_t+1, hdg, spd  -> sonercoleri pimrc


#############################################
### 2*pdoa, L -> roberts (parallel)

def pdoa_roberts_positioning(deld_L, deld_R, L):
    #following notations such as Y_A, D, A, and B are in line with the paper itself
    Y_A = L
    D   = L

    #calculate x,y position of the leading vehicle using eqs. in Robert's method
    if ((np.abs(deld_L) > 1e-4) and (np.abs(deld_R) > 1e-4)):
        A = Y_A ** 2 * (1 / (deld_L ** 2) - 1 / (deld_R ** 2))

        B1 = (-(Y_A ** 3) + 2 * (Y_A ** 2) * D + Y_A * (deld_L ** 2)) / (deld_L ** 2)
        B2 = (-(Y_A ** 3) + Y_A * (deld_R ** 2)) / (deld_R ** 2)
        B = B1 - 2 * D - B2

        C1 = ((Y_A ** 4) + 4 * (D ** 2) * (Y_A ** 2) + (deld_L ** 4) - 4 * D * (Y_A ** 3) - 2 * (Y_A ** 2) * (deld_L ** 2) + 4 * D * Y_A * (deld_L ** 2)) / (4 * (deld_L ** 2))
        C2 = ((Y_A ** 4) + (deld_R ** 4) - 2 * (Y_A ** 2) * (deld_R ** 2)) / (4 * (deld_R ** 2))
        C = C1 - D ** 2 - C2

        if deld_L * deld_R > 0:
            Y_B = (- B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
        else:
            Y_B = (- B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
        if ( ((Y_A ** 2 - 2 * Y_A * Y_B - deld_R ** 2) / (2 * deld_R)) ** 2 - (Y_B ** 2) < 0 ):
            # since assumes parallel, fails to find close delays -> negative in srqt
            out = (np.nan, np.nan, np.nan, np.nan)
        X_A = - np.sqrt(((Y_A ** 2 - 2 * Y_A * Y_B - deld_R ** 2) / (2 * deld_R)) ** 2 - (Y_B ** 2))
    elif (np.abs(deld_L) <= 1e-4):
        Y_B = Y_A / 2 - D
        if ((2 * D * Y_A - deld_R ** 2) / (2 - deld_R)) ** 2 - (D - Y_A / 2) ** 2 < 0:
            # since assumes parallel, fails to find close delays -> negative in srqt
            out = (np.nan, np.nan, np.nan, np.nan)
        X_A = - np.sqrt(((2 * D * Y_A - deld_R ** 2) / (2 - deld_R)) ** 2 - (D - Y_A / 2) ** 2)
    else:
        Y_B = Y_A / 2
        if ((2 * Y_A * D + deld_L ** 2) / (2 * deld_L)) ** 2 - (D + Y_A / 2) ** 2 < 0:
            # since assumes parallel, fails to find close delays -> negative in srqt
            out = (np.nan, np.nan, np.nan, np.nan)
        X_A = - np.sqrt(((2 * Y_A * D + deld_L ** 2) / (2 * deld_L)) ** 2 - (D + Y_A / 2) ** 2)

    # denklemler elec491 convention'ına göre yazıldığı için bir sign ve transpose operasyonu var
    x_txL = (0-Y_B)
    y_txL = -X_A
    x_txR = (0-Y_B) + L
    y_txR = -X_A # see the parallel assumption here

    return x_txL, y_txL, x_txR, y_txR 

#############################################
### 2*rtof, L -> bechadergue
def rtof_bechadergue_positioning(dL, dR, L):
    qt = dR**2 - L**2
    ### vector version
    #signer = -1*(qt > dL**2)
    #signer[signer == 0] = 1 
    signer = 2*(qt < dL**2)-1
    
    lng = np.sqrt(-dL**4 + (2*dL**2) * (dR**2 + L**2) - (dR**2 - L**2)**2)/(2*L)
    lat = signer*np.sqrt(dL**2 - lng**2)
    return lng, lat

#############################################
### 2*rtof, L -> xu
