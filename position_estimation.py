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
