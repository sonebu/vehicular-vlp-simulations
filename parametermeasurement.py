import numpy as np
from numba import njit
import time
import scipy as sp
from scipy import signal

#######################################################################################
### aoa measurement  -> sonercoleri
def measure_bearing_angle_fQRX(fqrx_samples, actual_phi):
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

def measure_bearing(sigA_buffer, sigB_buffer, sigC_buffer, sigD_buffer, wav_buffer, f_QRX, thd):
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
        aoa = 0.0001; # arbitrary value
    else:
        phi_hrz_est = ((qC+qD)-(qA+qB))/qpwr;
        aoa = measure_bearing_angle_fQRX(f_QRX, phi_hrz_est);

    return aoa


#######################################################################################
### range measurement -> bechadergue

### It's not really feasible to use the bechadergue range meas function as is during the simulations, 
### i.e., without pre-computing what can be precomputed, because it takes ages to simulate + hogs memory
###
### Therefore, we implement parts of this using the njit'ted functions below within the notebooks
### and precompute what can be precomputed there.
###
###def measure_range_bechadergue(sqr1, sqr2, sqrh, sqr_adc_s_gate, adc_re, f_adc_clock, c, r, N, f_e):
###    ### DUMP: bechadergue's hardware uses a "high-speed comparator", no details, 
###    ###       but it seems like this is NOT a hysteresis device since there's no mention of 
###    ###       either a debouncing function/circuit or a hysteresis threshold
###    ###       note -> the hysteresis used in VLR filtering is not used in HW AFAICS.
###    ###       check: "Vehicle-to-Vehicle Visible Light Phase-Shift Rangefinder Based on the Automotive Lighting"
###    ###
###    ###       Therefore, the signals below just accept >0 zero crossing detected signals
###    ###
###    s_1h      = dflipflop_vec(sqr1, sqrh)[adc_re]
###    s_2h      = dflipflop_vec(sqr2, sqrh)[adc_re]
###    s_phi     = np.logical_xor(s_1h, s_2h);
###    s_phi_pp  = s_phi*sqr_adc_s_gate
###    count     = counter_simulation_vec(s_phi_pp, s_gate)
###    f_i       = f_e/(r+1);
###    phase_shift_est = 2*np.pi*(np.asarray(count)*f_i/(N*f_adc_clock))
###    d_est = c*(phase_shift_est/(2*np.pi*2*f_e))
###    return d_est

# custom
@njit(parallel=True, fastmath=True)
def measure_range_bechadergue_dflipflop(inp_x, clock):
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
@njit(parallel=True, fastmath=True)
def measure_range_bechadergue_counter(signal, gate):
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

#######################################################################################
### range measurement -> roberts
### In contrast with the bechadergue method, this method can be used as is since the 
### signals will be clocked at say 10 MHz or something like that, not 10 GHz 
### like the square waves used in bechadergue's method
def measure_range_roberts(signal1, signal2, c, f_e):
    signal1 = sp.fft.fft(signal1);
    signal1[0:int(signal1.shape[0]/2)] = 0;
    signal1 = sp.fft.ifft(signal1);
    signal2 = sp.fft.fft(signal2);
    signal2[0:int(signal2.shape[0]/2)] = 0;
    signal2 = sp.fft.ifft(signal2);
    phase_shift_diff_est = np.mean(np.angle(signal1 * np.conjugate(signal2)));
    d_est = c*(phase_shift_diff_est/(2*np.pi*2*f_e))
    return d_est

#######################################################################################
### other

def generate_clocks(c, f_e, N, r, f_simulation, f_adc_clock, f_dig_clock):
    x_sim_spatial = c/f_simulation; # [m]

    f_gate       = f_e*(1/(N*(r+1)))
    t_gate_pulse = (1/f_gate)/2
    t_sim_stop   = t_gate_pulse*1.05; # stretch by 5 percent to ensure capturing gate pulse 

    t_sim_dt     = 1/f_simulation; # [s] , simulation clock period
    t_sim_start  = t_sim_dt; # [s], t_simulation rather than 0 avoids artifacts and sim ends at t_sim_stop this way
    sim_length = int(t_sim_stop/t_sim_dt)

    s_simulation   = np.linspace(t_sim_start, t_sim_stop, sim_length)
                                 
    t_adc_clock  = 1/f_adc_clock;   # [s] , measurement clock period

    simclock_subsample_rate = int(f_simulation/f_adc_clock)

    s_adc_clock = np.zeros_like(s_simulation)
    for i in range(0, int(simclock_subsample_rate/2)):
        s_adc_clock[ i::simclock_subsample_rate] = 1;

    s_adc_clock_lead  = s_adc_clock[1:]
    s_adc_clock_lag   = s_adc_clock[0:-1]
    s_adc_clock_re    = np.concatenate((np.asarray([False]), np.logical_and((1-s_adc_clock_lag), s_adc_clock_lead)))
    del s_adc_clock, s_adc_clock_lead, s_adc_clock_lag

    t_dig_clock  = 1/f_dig_clock;   # [s] , measurement clock period

    simclock_subsample_rate = int(f_simulation/f_dig_clock)

    s_dig_clock = np.zeros_like(s_simulation)
    for i in range(0, int(simclock_subsample_rate/2)):
        s_dig_clock[ i::simclock_subsample_rate] = 1;

    s_dig_clock_lead  = s_dig_clock[1:]
    s_dig_clock_lag   = s_dig_clock[0:-1]
    s_dig_clock_re    = np.concatenate((np.asarray([False]), np.logical_and((1-s_dig_clock_lag), s_dig_clock_lead)))
    del s_dig_clock, s_dig_clock_lead, s_dig_clock_lag


    return t_sim_dt, t_sim_stop, x_sim_spatial, s_simulation, s_adc_clock_re, s_dig_clock_re
