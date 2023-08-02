
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

        ### channel parameters
        self.weather        = weather        # one of "clear", "rain", "fog"
        self.temperature    = temperature    # [K], kelvin
        self.daynight       = daynight       # one of "night", "day_directsun", "day_indirectsun"
        self.rxconfig       = rxconfig       # npz file from the optics/ folder
        self.txconfig       = txconfig       # npz file from the optics/ folder
        self.istxlambertian = istxlambertian # True/False. Note that this needs to match txconfig

        # based on experimental data from related work, see references in Section II
        self.weather_attenuation_factors = dict()
        self.weather_attenuation_factors['clear'] = 0.0    # dB/m
        self.weather_attenuation_factors['rain']  = -0.05  # dB/m
        self.weather_attenuation_factors['fog']   = -0.2   # dB/m

        # based on experimental data from related work, see references in Section II
        self.daynight_noise_factors = dict()
        self.daynight_noise_factors['day_directsun']   = 1.000 # 5100 uA
        self.daynight_noise_factors['day_indirectsun'] = 0.145 # 740 uA
        self.daynight_noise_factors['night']           = 0.010 # very small

        # Original bandwidth = 10 MHz, we assume a 100 kHz BPF here like bechadergue, effectively reducing BW by 100x. 
        # See noise variance equations for further info on this
        self.bwscaling      = 0.01 # float. 
        
        ### simulation timing parameters
        self.c              = 299702547  # [m/s] speed of light
        self.f_adc_clk      = f_adc_clk  # [Hz] ADC clock freq, this is the sampling rate for the received signals
        self.f_e            = f_e        # [Hz] frequency of the emmitted wave from TX
        self.f_sim          = f_sim      # [Hz]

        ### bechadergue-specific configuration, only used for the auto-digital bearing measurement method
        self.autodig_f_clk = 1.0e8    # [Hz] digital clock freq
        self.autodig_r     = 5000     # unitless, heterodyning factor
        self.autodig_N     = 4        # unitless, averaging factor 

