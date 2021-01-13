# In[1]:
from VLP_methods.aoa import *
from VLP_methods.rtof import *
from VLP_methods.tdoa import *
from Bound_Estimation.matfile_read import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from scipy.io import loadmat
import math

# coding: utf-8

## In[1]:
# 'v2lcRun_sm1_laneChange'
# 'v2lcRun_sm2_platoonFormExit'
# 'v2lcRun_sm3_comparisonSoA'

data_names = ['v2lcRun_sm2_platoonFormExit']  # ['v2lcRun_sm1_laneChange', 'v2lcRun_sm2_platoonFormExit', 'v2lcRun_sm3_comparisonSoA']
folder_names = ['2/']  # ['1/', '2/', '3/']
# data_name = sys.argv[1]
#print(data_name)
for tr in range(50):
    print(tr)
    for i in range(len(data_names)):
        data_name = data_names[i]
        data_dir = 'SimulationData/' + data_name + '.mat'
        data = loadmat(data_dir)
        folder_name = folder_names[i]
        dp = 1
        data_point = '1000_point/'
        import cProfile, pstats
        profiler = cProfile.Profile()
        profiler.enable()

        data = load_mat(data_dir)

        max_power = data['tx']['power']
        area = data['qrx']['f_QRX']['params']['area']
        rx_radius = math.sqrt(area) / math.pi
        c = 3e8
        rx_fov = 50  # angle
        tx_half_angle = 60  # angle
        signal_freq = 1e6
        measure_dt = 1 / 2.5e6  # 2.5 MHz measure frequency

        time_ = data['vehicle']['t']['values']
        time_ = time_[::dp]
        vehicle_dt = data['vehicle']['t']['dt']

        rel_hdg = data['vehicle']['target_relative']['heading'][::dp]

        L_tgt = data['vehicle']['target']['width']
        L_ego = data['vehicle']['ego']['width']

        tgt_tx1_x = -1 * data['vehicle']['target_relative']['tx1_qrx4']['y'][::dp]
        tgt_tx1_y = data['vehicle']['target_relative']['tx1_qrx4']['x'][::dp]
        tgt_tx2_x = -1 * data['vehicle']['target_relative']['tx2_qrx3']['y'][::dp]
        tgt_tx2_y = data['vehicle']['target_relative']['tx2_qrx3']['x'][::dp]

        ego_qrx1_x = np.zeros(len(tgt_tx1_x))
        ego_qrx1_y = np.zeros(len(tgt_tx1_x))
        ego_qrx2_x = np.zeros(len(tgt_tx1_x))
        ego_qrx2_y = np.zeros(len(tgt_tx1_x)) + L_ego

        # delay parameters
        delay_11 = data['channel']['qrx1']['delay']['tx1'][::dp]
        delay_12 = data['channel']['qrx1']['delay']['tx2'][::dp]
        delay_21 = data['channel']['qrx2']['delay']['tx1'][::dp]
        delay_22 = data['channel']['qrx2']['delay']['tx2'][::dp]

        # received power of QRXes
        pow_qrx1_tx1 = np.array([data['channel']['qrx1']['power']['tx1']['A'][::dp], data['channel']['qrx1']['power']['tx1']['B'][::dp],
                                 data['channel']['qrx1']['power']['tx1']['C'][::dp],
                                 data['channel']['qrx1']['power']['tx1']['D'][::dp]])
        pow_qrx1_tx2 = np.array([data['channel']['qrx1']['power']['tx2']['A'][::dp], data['channel']['qrx1']['power']['tx2']['B'][::dp],
                                 data['channel']['qrx1']['power']['tx2']['C'][::dp],
                                 data['channel']['qrx1']['power']['tx2']['D'][::dp]])
        pow_qrx2_tx1 = np.array([data['channel']['qrx2']['power']['tx1']['A'][::dp], data['channel']['qrx2']['power']['tx1']['B'][::dp],
                                 data['channel']['qrx2']['power']['tx1']['C'][::dp],
                                 data['channel']['qrx2']['power']['tx1']['D'][::dp]])
        pow_qrx2_tx2 = np.array([data['channel']['qrx2']['power']['tx1']['A'][::dp], data['channel']['qrx2']['power']['tx1']['B'][::dp],
                                 data['channel']['qrx2']['power']['tx1']['C'][::dp],
                                 data['channel']['qrx2']['power']['tx1']['D'][::dp]])

        # noise params
        T = 298  # Kelvin
        I_bg = 750e-6  # 750 uA
        p_r_factor = data['qrx']['tia']['shot_P_r_factor']
        i_bg_factor = data['qrx']['tia']['shot_I_bg_factor']
        t_factor1 = data['qrx']['tia']['thermal_factor1']
        t_factor2 = data['qrx']['tia']['thermal_factor1']

        x, y, x_pose, y_pose, x_roberts, y_roberts, x_becha, y_becha = np.zeros((len(tgt_tx1_x), 2)), np.zeros((len(tgt_tx1_x),
                                                                                                                2)), \
                                                                       np.zeros((len(tgt_tx1_x), 2)), np.zeros((len(tgt_tx1_x),
                                                                                                                2)), \
                                                                       np.zeros((len(tgt_tx1_x), 2)), np.zeros((len(tgt_tx1_x),
                                                                                                                2)), \
                                                                       np.zeros((len(tgt_tx1_x), 2)), np.zeros((len(tgt_tx1_x),
                                                                                                                2))

        aoa = AoA(a_m=max_power, f_m1=signal_freq, f_m2=2*signal_freq, measure_dt=measure_dt, vehicle_dt=vehicle_dt,
                  w0=500, hbuf=1000, car_dist=L_tgt, fov=rx_fov)
        rtof = RToF(a_m=max_power, f_m=signal_freq, measure_dt=5e-9, vehicle_dt=vehicle_dt, car_dist=L_tgt,
                    r=499, N=1, c=c)
        tdoa = TDoA(a_m=max_power, f_m1=signal_freq, f_m2=signal_freq, measure_dt=measure_dt, vehicle_dt=vehicle_dt,
                    car_dist=L_tgt)
        for i in range(len(tgt_tx1_x)):
            # updating the given coordinates
            print("Iteration #", i, ": ")
            x[i] = (tgt_tx1_x[i], tgt_tx2_x[i])
            y[i] = (tgt_tx1_y[i], tgt_tx2_y[i])

            # providing the environment to methods
            delays = np.array([[delay_11[i], delay_21[i]], [delay_12[i], delay_22[i]]])
            H_q = np.array([[pow_qrx1_tx1[:, i], pow_qrx2_tx1[:, i]], [pow_qrx1_tx2[:, i], pow_qrx2_tx2[:, i]]])
            H = np.array([[np.sum(pow_qrx1_tx1[:, i]), np.sum(pow_qrx2_tx1[:, i])],
                         [np.sum(pow_qrx1_tx2[:, i]), np.sum(pow_qrx2_tx2[:, i])]])

            p_r1, p_r2, p_r3, p_r4 = H[0][0], H[0][1], H[1][0], H[1][1]
            remaining_factor = I_bg * i_bg_factor + T * (t_factor1 + t_factor2)
            noise_var1 = p_r1 * p_r_factor + remaining_factor
            noise_var2 = p_r2 * p_r_factor + remaining_factor
            noise_var3 = p_r3 * p_r_factor + remaining_factor
            noise_var4 = p_r4 * p_r_factor + remaining_factor
            noise_variance = np.array([[noise_var1, noise_var2], [noise_var3, noise_var4]])

            # noise_variance = np.array([[0.0, 0.0], [0.0, 0.0]])
            rem_fact_soner = I_bg * i_bg_factor + T * (t_factor1 + t_factor2 / 16)
            noise_var1_soner = np.array([H_q[0][0][0] * p_r_factor + rem_fact_soner, H_q[0][0][1] * p_r_factor + rem_fact_soner,
                                   H_q[0][0][2] * p_r_factor + rem_fact_soner, H_q[0][0][3] * p_r_factor + rem_fact_soner])
            noise_var2_soner = np.array([H_q[0][1][0] * p_r_factor + rem_fact_soner, H_q[0][1][1] * p_r_factor + rem_fact_soner,
                                   H_q[0][1][2] * p_r_factor + rem_fact_soner, H_q[0][1][3] * p_r_factor + rem_fact_soner])
            noise_var3_soner = np.array([H_q[1][0][0] * p_r_factor + rem_fact_soner, H_q[1][0][1] * p_r_factor + rem_fact_soner,
                                   H_q[1][0][2] * p_r_factor + rem_fact_soner, H_q[1][0][3] * p_r_factor + rem_fact_soner])
            noise_var4_soner = np.array([H_q[1][1][0] * p_r_factor + rem_fact_soner, H_q[1][1][1] * p_r_factor + rem_fact_soner,
                                   H_q[1][1][2] * p_r_factor + rem_fact_soner, H_q[1][1][3] * p_r_factor + rem_fact_soner])
            noise_variance_soner = np.array([[noise_var1_soner, noise_var2_soner], [noise_var3_soner, noise_var4_soner]])

            # making estimations
            tx_aoa = aoa.estimate(delays=delays, H_q=H_q, noise_variance=noise_variance_soner)
            print("AoA finished")
            tx_rtof = rtof.estimate(all_delays=delays, H=H, noise_variance=noise_variance)
            print("RToF finished")
            print("heading:", rel_hdg[i])
            tx_tdoa = tdoa.estimate(delays=delays, H=H, noise_variance=noise_variance)
            print("TDoA finished")

            # storing to plot later
            x_pose[i] = tx_aoa[0]
            y_pose[i] = tx_aoa[1]
            x_becha[i] = tx_rtof[0]
            y_becha[i] = tx_rtof[1]
            x_roberts[i] = tx_tdoa[0]
            y_roberts[i] = tx_tdoa[1]

        y_data = np.copy(y)
        x_data = np.copy(x)
        for i in range(len(y)):
            y_data[i] = (y[i][0] + y[i][1])/2
            x_data[i] = (x[i][0] + x[i][1])/2

        np.savetxt('GUI_data/'+data_point+folder_name+'x.txt', x, delimiter=',')
        np.savetxt('GUI_data/'+data_point+folder_name+'x_pose.txt', x_pose, delimiter=',')
        np.savetxt('GUI_data/'+data_point+folder_name+'x_becha.txt', x_becha, delimiter=',')
        np.savetxt('GUI_data/'+data_point+folder_name+'x_roberts.txt', x_roberts, delimiter=',')
        np.savetxt('GUI_data/'+data_point+folder_name+'x_data.txt', x_data, delimiter=',')
        np.savetxt('GUI_data/'+data_point+folder_name+'y.txt', y, delimiter=',')
        np.savetxt('GUI_data/'+data_point+folder_name+'y_data.txt', y_data, delimiter=',')
        np.savetxt('GUI_data/'+data_point+folder_name+'/y_becha.txt', y_becha, delimiter=',')
        np.savetxt('GUI_data/'+data_point+folder_name+'y_roberts.txt', y_roberts, delimiter=',')
        np.savetxt('GUI_data/'+data_point+folder_name+'y_pose.txt', y_pose, delimiter=',')
        np.savetxt('GUI_data/'+data_point+folder_name+'time.txt', time_, delimiter=',')
        np.savetxt('GUI_data/'+data_point+folder_name+'rel_hdg.txt', rel_hdg, delimiter=',')
        print(time_)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('ncalls')
        # stats.print_stats()



        #axs[0].plot(t1, f(t1), 'o', t2, f(t2), '-')