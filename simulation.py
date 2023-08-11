import numpy as np

##############################################################################
### utility functions
def map_rx_config(input_dict):
    f_QRX    = input_dict['f_QRX']
    pd_snst  = input_dict['tia_gamma']
    pd_gain  = input_dict['tia_R_F']
    pd_dim   = input_dict['detecting_area'] # square detector dimension in milimeters. needs a /1000 in calcs
    rx_P_rx_factor     = input_dict['tia_shot_P_r_factor']
    rx_I_bg_factor     = input_dict['tia_shot_I_bg_factor']
    rx_thermal_factor1 = input_dict['tia_thermal_factor1']
    rx_thermal_factor2 = input_dict['tia_thermal_factor2']

    return f_QRX, pd_snst, pd_gain, pd_dim, rx_P_rx_factor, rx_I_bg_factor, rx_thermal_factor1, rx_thermal_factor2

def map_tx_config(input_dict):
    tx_ha   = input_dict['half_angle']
    tx_pwr  = input_dict['power']
    tx_norm = input_dict['normalization_factor']
    tx_lambertian_order = int(-np.log(2)/np.log(np.cos(np.deg2rad(tx_ha))));

    return tx_ha, tx_pwr, tx_norm, tx_lambertian_order

def map_tx_config_nonlambertian(input_dict):
    tx_pattern    = input_dict['pattern']
    tx_phiarray   = input_dict['phi_array']
    tx_thetaarray = input_dict['theta_array']
    tx_pwr        = input_dict['power']
    tx_norm       = input_dict['normalization_factor']
    
    return tx_pattern, tx_phiarray, tx_thetaarray, tx_pwr, tx_norm

def generate_clocksampler_from_s_simulation(f_desired_clock, f_simulation, s_simulation):
    t_desired_clock         = 1/f_desired_clock; # [s], not used, just to demonstrate units for varibles 
    simclock_subsample_rate = int(f_simulation/f_desired_clock)
    
    # set a tick every 2 samples (since that's what a clock is)
    s_desired_clock         = np.zeros_like(s_simulation)
    for i in range(0, int(simclock_subsample_rate/2)):
        s_desired_clock[ i::simclock_subsample_rate] = 1;

    # get a rising edge signal
    s_desired_clock_lead  = s_desired_clock[1:]
    s_desired_clock_lag   = s_desired_clock[0:-1]
    s_desired_clock_re    = np.concatenate((np.asarray([False]), np.logical_and((1-s_desired_clock_lag), s_desired_clock_lead)))
    del s_desired_clock, s_desired_clock_lead, s_desired_clock_lag

    # the rising edge signal is the desired clocksampler with which we will sample other signals like signal_to_sample[s_desired_clock_re]
    # signal_to_sample and s_desired_clock_re are of size len(s_simulation), that's why this works
    return s_desired_clock_re

def generate_simulation_clocks(t_sim_stop, f_simulation, f_desired_clock, gen_clocked_only=False):
    if(gen_clocked_only):
        t_sim_dt           = 1/f_desired_clock; # [s] , simulation clock period
        t_sim_start        = t_sim_dt; # [s], t_simulation rather than 0 avoids artifacts and sim ends at t_sim_stop this way
        sim_length         = int(t_sim_stop/t_sim_dt)
        s_desired_clock_re = np.linspace(t_sim_start, t_sim_stop, sim_length)

        # NOTE that f_simulation is ignored in this path, and there's only 1 clock output

        return s_desired_clock_re
    else:
        t_sim_dt     = 1/f_simulation; # [s] , simulation clock period
        t_sim_start  = t_sim_dt; # [s], t_simulation rather than 0 avoids artifacts and sim ends at t_sim_stop this way
        sim_length   = int(t_sim_stop/t_sim_dt)
        s_simulation = np.linspace(t_sim_start, t_sim_stop, sim_length)
                            
        s_desired_clock_re = generate_clocksampler_from_s_simulation(f_desired_clock, f_simulation, s_simulation)

        return s_simulation, s_desired_clock_re


##############################################################################
### Configuration object definition
class Simulation:
    def __init__(self, weather        = "clear", 
                       temperature    = 298, 
                       daynight       = "night", 
                       rxconfig       = "optics/qrx_planoconvex.npz", 
                       txconfig       = "optics/tx_lambertian_20deg_2W.npz", 
                       istxlambertian = True, 
                       f_adc_clk      = 1.0e7, 
                       f_e            = 1.0e6, 
                       f_sim          = 1.0e10):

        # channel parameters
        self.weather        = weather        # one of "clear", "rain", "fog"
        self.temperature    = temperature    # [K], kelvin
        self.daynight       = daynight       # one of "night", "day_directsun", "day_indirectsun"
        self.rxconfig       = rxconfig       # npz file from the optics/ folder
        self.txconfig       = txconfig       # npz file from the optics/ folder
        self.istxlambertian = istxlambertian # True/False. Note that this needs to match txconfig

        # based on experimental data from related work, see references in Section II
        weather_attenuation_factors = dict()
        weather_attenuation_factors['clear'] = 0.0    # dB/m
        weather_attenuation_factors['rain']  = -0.05  # dB/m
        weather_attenuation_factors['fog']   = -0.2   # dB/m
        self.attenuation_factor = weather_attenuation_factors[weather];

        # based on experimental data from related work, see references in Section II
        daynight_noise_factors = dict()
        daynight_noise_factors['day_directsun']   = 1.000 # 5100 uA
        daynight_noise_factors['day_indirectsun'] = 0.145 # 740 uA
        daynight_noise_factors['night']           = 0.010 # very small
        
        # minmax bounds to be safe if someone sets the noise factor themselves
        bg_current = (np.minimum(1, np.maximum(0, daynight_noise_factors[daynight]))*5100)*1e-6;

        # load receiver config
        a = np.load(rxconfig);
        f_QRX, pd_snst, pd_gain, pd_dim, rx_P_rx_factor, rx_I_bg_factor, rx_thermal_factor1, rx_thermal_factor2 = map_rx_config(a);

        # Original bandwidth = 10 MHz, we assume a 100 kHz BPF here like bechadergue, effectively reducing BW by 100x. 
        # See noise variance equations for further info on this
        self.bwscaling = 0.01 # float
        
        self.rx_config_bundle = dict()
        self.rx_config_bundle["f_QRX"] = f_QRX
        self.rx_config_bundle["pd_sensitivity"] = pd_snst
        self.rx_config_bundle["pd_gain"] = pd_gain
        self.rx_config_bundle["pd_dimension"] = pd_dim
        self.rx_config_bundle["rx_P_rx_factor"] = rx_P_rx_factor
        self.rx_config_bundle["rx_I_bg_factor"] = rx_I_bg_factor
        self.rx_config_bundle["rx_thermal_factor1"] = rx_thermal_factor1
        self.rx_config_bundle["rx_thermal_factor2"] = rx_thermal_factor2

        # this factor is precomputed since it's the same for all links (/16 due to $C_i^2$ in the thermal_factor2, each cell gets 1/4 of the total cap)
        self.rx_config_bundle["thermal_and_bg_curr"] = rx_I_bg_factor*self.bwscaling*bg_current + temperature*(self.bwscaling*rx_thermal_factor1 + (self.bwscaling**3)*rx_thermal_factor2)

        # load transmitter config
        if(self.istxlambertian):
            a = np.load(txconfig)
            tx_ha, tx_pwr, tx_norm, tx_lambertian_order = map_tx_config(a);
            self.tx_config_bundle = dict()
            self.tx_config_bundle["istxlambertian"] = istxlambertian
            self.tx_config_bundle["tx_halfangle"] = tx_ha
            self.tx_config_bundle["tx_lambertian_order"] = tx_lambertian_order
            self.tx_config_bundle["tx_pwr"] = tx_pwr
            self.tx_config_bundle["tx_norm"] = tx_norm
        else:
            a = np.load(txconfig)
            tx_pattern, tx_phiarray, tx_thetaarray, tx_pwr, tx_norm = map_tx_config_nonlambertian(a);
            self.tx_config_bundle = dict()
            self.tx_config_bundle["tx_pattern"] = tx_pattern
            self.tx_config_bundle["tx_phiarray"] = tx_phiarray
            self.tx_config_bundle["tx_thetaarray"] = tx_thetaarray
            self.tx_config_bundle["tx_pwr"] = tx_pwr
            self.tx_config_bundle["tx_norm"] = tx_norm

        ### simulation timing parameters
        self.lightspeed     = 299702547  # [m/s] speed of light
        self.f_adc_clock    = f_adc_clk  # [Hz] ADC clock freq, this is the sampling rate for the received signals
        self.f_emitted      = f_e        # [Hz] frequency of the emmitted wave from TX
        self.f_simulation   = f_sim      # [Hz] simulation master clock freq

