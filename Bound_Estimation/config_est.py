from yacs.config import CfgNode as CN
from matfile_read import load_mat

bound_est_data = CN()
# theoretical analysis parameters
bound_est_data.params = CN()
# points skipped while reading data
bound_est_data.params.number_of_skip_data = 10
# noise params
bound_est_data.params.T = 298  # Kelvin
bound_est_data.params.I_bg = 750e-6  # 750 uA
# angle params
bound_est_data.params.rx_fov = 50  # angle
bound_est_data.params.tx_half_angle = 60  # angle
# signal params
bound_est_data.params.signal_freq = 1e6  # 1 MHz signal frequency
bound_est_data.params.measure_dt = 1 / 2.5e6  # 2.5 MHz measure frequency

# plotting parameters
bound_est_data.plot = CN()
# source directory
bound_est_data.plot.directory = '../GUI_data/100_point_'