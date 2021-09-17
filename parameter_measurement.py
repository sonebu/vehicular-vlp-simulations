import numpy as np
from numba import njit
import time

#######################################################################################
### aoa measurement  -> sonercoleri
def measure_angle_fQRX(fqrx_samples, actual_phi):
    fqrx_values   = fqrx_samples[:,1]
    diff_value    = np.abs(fqrx_values - actual_phi)
    argpart_value = np.argpartition(diff_value, 2) # , axis = 0

    closest_value_0_idx = argpart_value[0] # index for closest y value in LUT
    closest_value_1_idx = argpart_value[1] # index for 2nd closest y value in LUT

    closest_value_0 = fqrx_values[closest_value_0_idx] # closest y value in LUT
    closest_value_1 = fqrx_values[closest_value_1_idx] # 2nd closest y value in LUT

    fqrx_angles = fqrx_samples[:,0];
    closest_angle_0 = fqrx_angles[closest_value_0_idx] # angle value for closest y value in LUT
    closest_angle_1 = fqrx_angles[closest_value_1_idx] # angle value for 2nd closest y value in LUT

    value_dist_to_0 = actual_phi - closest_value_0; # increment

    angle_interval = closest_angle_1 - closest_angle_0;  # total span between two angle values
    value_interval = closest_value_1 - closest_value_0; # total span between two y values
    slope          = angle_interval / value_interval;
    angle_est      = value_dist_to_0 * slope + closest_angle_0;
    return angle_est

def aoa_measurement(sigA_buffer, sigB_buffer, sigC_buffer, sigD_buffer, wav_buffer, f_QRX, thd):
    qA = np.abs(np.mean(sigA_buffer*wav_buffer));
    qB = np.abs(np.mean(sigB_buffer*wav_buffer));
    qC = np.abs(np.mean(sigC_buffer*wav_buffer));
    qD = np.abs(np.mean(sigD_buffer*wav_buffer));
    qpwr = qA + qB + qC + qD;
    
    # When the target is out of the FoV, obviously it's an invalid result.
    # we detect this here and assign a NaN to those results outside
    # so that they don't appear on the plot. It's just extra clutter when
    # they're plotted.
    if( ((qC + qD) < thd) or ((qA + qB) < thd) ):
        aoa = 0;
    else:
        phi_hrz_est = ((qC+qD)-(qA+qB))/qpwr;
        aoa = measure_angle_fQRX(f_QRX, phi_hrz_est);

    return aoa

#######################################################################################
### pdoa measurement -> roberts

def pdoa_deld_measure(rxL, rxR):
    rxL = np.fft.fft(rxL);
    rxL[0:int(rxL.shape[0]/2)] = 0;
    rxL = np.fft.ifft(rxL);
    rxR = np.fft.fft(rxR);
    rxR[0:int(rxR.shape[0]/2)] = 0;
    rxR = np.fft.ifft(rxR);
    phase_shift_diff_est = np.mean(np.angle(rxL * np.conjugate(rxR)));
    return phase_shift_diff_est


#######################################################################################
### rtof measurement -> bechadergue

# taken from: https://stackoverflow.com/a/23291658
def hyst(x, th_lo, th_hi, initial = False):
    hi = x >= th_hi
    lo_or_hi = (x <= th_lo) | hi
    ind = np.nonzero(lo_or_hi)[0]
    if not ind.size: # prevent index error if ind is empty
        return np.zeros_like(x, dtype=bool) | initial
    cnt = np.cumsum(lo_or_hi) # from 0 to len(x)
    return np.where(cnt, hi[ind[cnt-1]], initial)

# custom
@njit(parallel=True, fastmath=True)
def dflipflop_vec(inp_x, clock):
    inp_lead = inp_x[1:]
    inp_lag  = inp_x[0:-1]
    clk_lead = clock[1:]
    clk_lag  = clock[0:-1]

    clk_re = np.concatenate((np.asarray([False]), np.logical_and((1-clk_lag),clk_lead)))
    out_pp = inp_x[clk_re]

    re_idx      = np.where(clk_re==1)
    re_idx_lead = re_idx[0][1:]
    re_idx_lag  = re_idx[0][0:-1]

    out_t = np.zeros(inp_x.shape)
    for i in range(0, len(re_idx_lag)):
        out_t[re_idx_lag[i]:re_idx_lead[i]] = out_pp[i]

    return out_t

# custom
def counter_simulation_vec(signal, gate):
    gate_lead = gate[1:]
    gate_lag  = gate[0:-1]
    gate_re   = np.concatenate((np.asarray([False]), np.logical_and(1-gate_lag, gate_lead)))
    gate_fe   = np.concatenate((np.asarray([False]), np.logical_and(gate_lag, 1-gate_lead)))
    
    re_idx = np.where(gate_re==1)[0];
    fe_idx = np.where(gate_fe==1)[0];

    count = []
    for i in range(0, len(re_idx)):
        count.append( np.sum( signal[ re_idx[i]:fe_idx[i] ] ) )

    return count

# custom (does a bit of debouncing, super slow)
def counter_simulation_iterative(signal, state_prev, state_ctr, count):
    if(   (signal[-1]==0) and (np.sum(signal[0:4])==0) ):
        state_curr = 0;
    elif( (signal[-1]==1) and (np.sum(signal[0:4])!=4) and (state_prev==0) ):
        state_curr = 1;
    elif( (signal[-1]==1) and (np.sum(signal[0:4])==4) ):
        state_curr = 2;
    elif( (signal[-1]==0) and (np.sum(signal[0:4])!=0) and (state_prev==2) ):
        state_curr = 3;
    else:
        state_curr = state_prev;

    # state ops:
    if(state_curr==2):
        state_ctr = state_ctr + 1; 
    elif(state_curr==1):
        state_ctr = state_ctr + 0.5;
    elif(state_curr==3):
        state_ctr = state_ctr + 0.5;
    elif(state_curr==0):
        pass
        #state_ctr = 0;
    
    state_prev = state_curr;
    
    return state_prev, state_ctr

@njit(parallel=True, fastmath=True)
def gen_s_h_sin(f_e, rtof_r, s_simulation, rr):
    s_h_sin    = np.sin(2*np.pi* f_e*(rtof_r/(rtof_r+1))*s_simulation - np.pi/32);
    s_h_zc_thd = (np.amax(s_h_sin) - np.amin(s_h_sin))/rr;
    return s_h_sin, s_h_zc_thd

@njit(parallel=True, fastmath=True)
def gen_s_e_sin(f_e, s_simulation, rr):
    s_e_sin    = np.sin(2*np.pi* f_e *s_simulation - np.pi/32);
    s_e_zc_thd = (np.amax(s_e_sin) - np.amin(s_e_sin))/rr;
    return s_e_sin, s_e_zc_thd 

@njit(parallel=True, fastmath=True)
def gen_s_gate_sin(f_e, rtof_r, rtof_N, s_simulation, rr):
    s_gate_sin = np.sin(2*np.pi* f_e*(1/(rtof_N*(rtof_r+1))) *s_simulation - np.pi/32);
    s_gate_zc_thd = (np.amax(s_gate_sin) - np.amin(s_gate_sin))/rr;
    return s_gate_sin, s_gate_zc_thd 

#@njit(parallel=True, fastmath=True)
def rtof_d_measure(s_simulation, s_adc_clock_re, rx, rtof_N, rtof_r, f_e, f_adc_clock, c, rr):
    #s_h_sin    = np.sin(2*np.pi* f_e*(rtof_r/(rtof_r+1))     *s_simulation - np.pi/32);
    #s_h_zc_thd = (np.amax(s_h_sin) - np.amin(s_h_sin))/rr;
    s_h_sin, s_h_zc_thd = gen_s_h_sin(f_e, rtof_r, s_simulation, rr);
    s_h        = hyst(s_h_sin, -s_h_zc_thd, s_h_zc_thd)
    #del s_h_sin

    #s_e_sin    = np.sin(2*np.pi* f_e               *s_simulation - np.pi/32);
    #s_e_zc_thd = (np.amax(s_e_sin) - np.amin(s_e_sin))/rr;
    s_e_sin, s_e_zc_thd = gen_s_e_sin(f_e, s_simulation, rr);
    s_e        = hyst(s_e_sin, -s_e_zc_thd, s_e_zc_thd)
    #del s_e_sin

    # burdan önce heralde bandpass filtre olcak, yoksa halay çeker bu alttaki hysteresis
    s_r_zc_thd = (np.amax(rx) - np.amin(rx))/rr;
    s_r        = hyst(rx, -s_r_zc_thd, s_r_zc_thd)
    #del rx
    
    s_eh = dflipflop_vec(s_e, s_h)[s_adc_clock_re]
    s_rh = dflipflop_vec(s_r, s_h)[s_adc_clock_re]
    #del s_e, s_h
    
    #s_gate_sin = np.sin(2*np.pi* f_e*(1/(rtof_N*(rtof_r+1))) *s_simulation - np.pi/32);
    #s_gate_zc_thd = (np.amax(s_gate_sin) - np.amin(s_gate_sin))/rr;
    s_gate_sin, s_gate_zc_thd = gen_s_gate_sin(f_e, rtof_r, rtof_N, s_simulation, rr);
    s_gate = hyst(s_gate_sin, -s_gate_zc_thd, s_gate_zc_thd)
    s_gate = s_gate[s_adc_clock_re]
    #del s_gate_sin

    s_phi    = np.logical_xor(s_eh, s_rh);
    #del s_rh, s_eh

    s_phi_pp  = s_phi*s_gate # clock applied implicitly
    #del s_phi

    count = counter_simulation_vec(s_phi_pp, s_gate)

    f_i = f_e/(rtof_r+1);
    phase_shift_est = 2*np.pi*(np.asarray(count)*f_i/(rtof_N*f_adc_clock))
    d_est = c*(phase_shift_est/(2*np.pi*2*f_e))

    return d_est