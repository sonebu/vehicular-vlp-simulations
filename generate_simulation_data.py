from VLP_methods.aoa import *
from VLP_methods.rtof import *
from VLP_methods.tdoa import *
from Bound_Estimation.matfile_read import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from scipy.io import loadmat
import math
import os
from config import gen_sim_data


def main():
    # obtain real-life simulation scenario data from specified folders
    data_names = gen_sim_data.names.data_names
    folder_names = gen_sim_data.names.folder_names

    # run for multiple iterations
    for itr in range(gen_sim_data.params.start_point_of_iter, gen_sim_data.params.end_point_of_iter):
        print(itr)
        for i in range(len(data_names)):
            data_name = data_names[i]
            data_dir = 'SimulationData/' + data_name + '.mat'
            data = loadmat(data_dir)

            folder_name = folder_names[i]
            dp = gen_sim_data.params.number_of_skip_data
            data_point = str(int(1000 / dp)) + '_point_' + str(itr) + '/'

            f_name = 'GUI_data/'+data_point+folder_name
            print(f_name)

            if not os.path.exists(f_name):
                os.makedirs(f_name)

            max_power = data['tx']['power']
            area = data['qrx']['f_QRX']['params']['area']
            rx_radius = math.sqrt(area) / math.pi

            c = gen_sim_data.params.c
            rx_fov = gen_sim_data.params.rx_fov
            tx_half_angle = gen_sim_data.params.tx_half_angle
            signal_freq = gen_sim_data.params.signal_freq
            measure_dt = gen_sim_data.params.measure_dt

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
            T = gen_sim_data.params.T
            I_bg = gen_sim_data.params.I_bg
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

            aoa = AoA(a_m=max_power, f_m1=signal_freq, f_m2=2*signal_freq, measure_dt=measure_dt, vehicle_dt=vehicle_dt * dp,
                      w0=gen_sim_data.params.w0, hbuf= int(vehicle_dt / measure_dt), car_dist=L_tgt, fov=rx_fov)
            rtof = RToF(a_m=max_power, f_m=signal_freq, measure_dt=gen_sim_data.params.rtof_measure_dt, vehicle_dt=vehicle_dt * dp, car_dist=L_tgt,
                        r=gen_sim_data.params.r, N=gen_sim_data.params.N , c=c)
            tdoa = TDoA(a_m=max_power, f_m1=signal_freq, f_m2=signal_freq, measure_dt=measure_dt , vehicle_dt=vehicle_dt * dp,
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

            np.savetxt(f_name+'x.txt', x, delimiter=',')
            np.savetxt(f_name+'x_pose.txt', x_pose, delimiter=',')
            np.savetxt(f_name+'x_becha.txt', x_becha, delimiter=',')
            np.savetxt(f_name+'x_roberts.txt', x_roberts, delimiter=',')
            np.savetxt(f_name+'x_data.txt', x_data, delimiter=',')
            np.savetxt(f_name+'y.txt', y, delimiter=',')
            np.savetxt(f_name+'y_data.txt', y_data, delimiter=',')
            np.savetxt(f_name+'y_becha.txt', y_becha, delimiter=',')
            np.savetxt(f_name+'y_roberts.txt', y_roberts, delimiter=',')
            np.savetxt(f_name+'y_pose.txt', y_pose, delimiter=',')
            np.savetxt(f_name+'time.txt', time_, delimiter=',')
            np.savetxt(f_name+'rel_hdg.txt', rel_hdg, delimiter=',')
            print(time_)


if __name__ == '__main__':
    main()