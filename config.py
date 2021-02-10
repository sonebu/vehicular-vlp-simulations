from yacs.config import CfgNode as CN

# Parameters for generate_simulation_data.py
gen_sim_data = CN()
gen_sim_data.names = CN()
# directory names
gen_sim_data.names.data_names = ['v2lcRun_sm1_laneChange', 'v2lcRun_sm2_platoonFormExit', 'v2lcRun_sm3_comparisonSoA']
gen_sim_data.names.folder_names = ['1/', '2/', '3/']
gen_sim_data.params = CN()
# data manipulation params
gen_sim_data.params.start_point_of_iter = 216
gen_sim_data.params.end_point_of_iter = 218
gen_sim_data.params.number_of_skip_data = 10
# environment params
gen_sim_data.params.c = 3e8
gen_sim_data.params.rx_fov = 50  # angle
gen_sim_data.params.tx_half_angle = 60  # angle
gen_sim_data.params.signal_freq = 1e6
gen_sim_data.params.measure_dt = 1 / 2.5e6  # 2.5 MHz measure frequency
# noise params
gen_sim_data.params.T = 298  # Kelvin
gen_sim_data.params.I_bg = 750e-6  # 750 uA
# aoa params
gen_sim_data.params.w0 = 0
# rtof params
gen_sim_data.params.rtof_measure_dt = 5e-9
gen_sim_data.params.r = 499
gen_sim_data.params.N = 1
# Parameters for plot_simulation_data.py
plot_sim_data = CN()
plot_sim_data.names = CN()
plot_sim_data.names.sm = [1, 2, 3]
plot_sim_data.names.folder_name = 'GUI_data/100_point_202/'
plot_sim_data.names.dir = 'GUI_data/100_point_202/'
plot_sim_data.names.data_names = gen_sim_data.names.data_names
plot_sim_data.names.folder_names = gen_sim_data.names.folder_names
# Parameters for simulation.py
sim_data = CN()
sim_data.names = CN()
sim_data.names.data_names = gen_sim_data.names.data_names
sim_data.names.folder_names = gen_sim_data.names.folder_names
sim_data.names.intro_img_dir_name = "Figure/intro_img.png"
sim_data.params = CN()
sim_data.params.size_width = 1200  # main window width
sim_data.params.size_height = 800  # main window height
sim_data.params.number_of_skip_data = 1
sim_data.params.img_ego_s_dir = 'red_racing_car_top_view_preview.png'
sim_data.params.img_tgt_s_dir = 'green_racing_car_top_view_preview.png'
sim_data.params.img_tgt_f_dir = 'green_racing_car_top_view_preview.png'
