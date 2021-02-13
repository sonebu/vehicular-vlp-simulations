from CRLB_init import *
from matfile_read import load_mat
from glob import glob
import math
import matplotlib.pyplot as plt
from config_est import bound_est_data

# Roberts' method, CRLB calculation for a single position estimation
def roberts_crlb_single_instance(crlb_obj, tx1, tx2, delays, curr_t, dt_vhc, max_pow, sig_freq, meas_dt, T, i_bg, noise_factors, powers):
    flag = False
    fim = np.zeros(shape=(2,2))

    for param1, param2 in zip(range(2), range(2)):
        for i in range(2):
            for j in range(2):
                ij = (i + 1)*10 + (j + 1)
                E_1, E_2, E_3 = signal_generator(curr_t, dt_vhc, max_pow, sig_freq, delays[i][j], meas_dt)
                h_ij = crlb_obj.get_h_ij(ij, tx1, tx2, flag)
                dh_dk1 = crlb_obj.get_d_hij_d_param(param1 + 1, ij, tx1, tx2, flag)
                dh_dk2 = crlb_obj.get_d_hij_d_param(param2 + 1, ij, tx1, tx2, flag)
                dtau_dk1 = crlb_obj.get_d_tau_d_param(param1 + 1, ij, tx1, tx2, flag)
                dtau_dk2 = crlb_obj.get_d_tau_d_param(param2 + 1, ij, tx1, tx2, flag)

                dh_dk1_dh_dk2 = dh_dk1 * dh_dk2
                h_dh_dk1_dtau_dk2 = - h_ij * dh_dk1 * dtau_dk2
                h_dh_dk2_dtau_dk1 = - h_ij * dh_dk2 * dtau_dk1
                hsq_dtau_dk1_dtau_dk2 = h_ij ** 2 * dtau_dk1 * dtau_dk2

                p_r = np.sum(powers[i][j])
                noise_effect = 1 / (p_r * noise_factors[0] + i_bg * noise_factors[1] + T * (noise_factors[2] + noise_factors[3]))

                fim[param1][param2] += noise_effect * (dh_dk1_dh_dk2 * E_2 \
                                       + (h_dh_dk1_dtau_dk2 + h_dh_dk2_dtau_dk1) * E_3 \
                                       + hsq_dtau_dk1_dtau_dk2 * E_1)
    return np.linalg.inv(fim)


#  Bechadergue's method, CRLB calculation for a single position estimation
def bechadergue_crlb_single_instance(crlb_obj, tx1, tx2, delays, curr_t, dt_vhc, max_pow, sig_freq, meas_dt, T, i_bg, noise_factors, powers):
    fim = np.zeros(shape=(4, 4))

    for param1, param2 in zip(range(4), range(4)):
        for i in range(2):
            for j in range(2):
                ij = (i + 1) * 10 + (j + 1)
                h_ij = crlb_obj.get_h_ij(ij, tx1, tx2)
                E_1, E_2, E_3 = signal_generator(curr_t, dt_vhc, max_pow, sig_freq, delays[i][j], meas_dt)
                dh_dk1 = crlb_obj.get_d_hij_d_param(param1 + 1, ij, tx1, tx2)
                dh_dk2 = crlb_obj.get_d_hij_d_param(param2 + 1, ij, tx1, tx2)
                dtau_dk1 = crlb_obj.get_d_tau_d_param(param1 + 1, ij, tx1, tx2)
                dtau_dk2 = crlb_obj.get_d_tau_d_param(param2 + 1, ij, tx1, tx2)

                dh_dk1_dh_dk2 = dh_dk1 * dh_dk2
                h_dh_dk1_dtau_dk2 = - h_ij * dh_dk1 * dtau_dk2
                h_dh_dk2_dtau_dk1 = - h_ij * dh_dk2 * dtau_dk1
                hsq_dtau_dk1_dtau_dk2 = h_ij ** 2 * dtau_dk1 * dtau_dk2

                p_r = np.sum(powers[i][j])
                noise_effect = 1 / (p_r * noise_factors[0] + i_bg * noise_factors[1] + T * (
                            noise_factors[2] + noise_factors[3]))

                fim[param1][param2] += noise_effect * (dh_dk1_dh_dk2 * E_2 \
                                       + (h_dh_dk1_dtau_dk2 + h_dh_dk2_dtau_dk1) * E_3 \
                                       + hsq_dtau_dk1_dtau_dk2 * E_1)
    return np.linalg.inv(fim)


#  Soner's method, CRLB calculation for a single position estimation
def soner_crlb_single_instance(crlb_obj, tx1, tx2, delays, curr_t, dt_vhc, max_pow, sig_freq, meas_dt, T, i_bg, noise_factors, powers):
    fim = np.zeros(shape=(4, 4))

    for param1, param2 in zip(range(4), range(4)):
        for i in range(2):
            for j in range(2):
                for qrx in range(4):
                    ij = (i + 1) * 10 + (j + 1)
                    q = qrx + 1
                    E_1, E_2, E_3 = signal_generator(curr_t, dt_vhc, max_pow, sig_freq, delays[i][j], meas_dt)

                    h_ijq = crlb_obj.get_h_ijq(ij, q, tx1, tx2)
                    dh_dk1 = crlb_obj.get_d_hij_q_d_param(param1 + 1, ij, q, tx1, tx2)
                    dh_dk2 = crlb_obj.get_d_hij_q_d_param(param2 + 1, ij, q, tx1, tx2)
                    dtau_dk1 = crlb_obj.get_d_tau_d_param(param1 + 1, ij, tx1, tx2)
                    dtau_dk2 = crlb_obj.get_d_tau_d_param(param2 + 1, ij, tx1, tx2)

                    dh_dk1_dh_dk2 = dh_dk1 * dh_dk2
                    h_dh_dk1_dtau_dk2 = h_ijq * dh_dk1 * dtau_dk2
                    h_dh_dk2_dtau_dk1 = h_ijq * dh_dk2 * dtau_dk1
                    hsq_dtau_dk1_dtau_dk2 = h_ijq ** 2 * dtau_dk1 * dtau_dk2

                    p_r = powers[i][j][qrx]
                    noise_effect = 1 / (p_r * noise_factors[0] + i_bg * noise_factors[1] + T * (
                                noise_factors[2] + noise_factors[3]/ 16))  # /16 comes from capacitance division

                    fim[param1][param2] += noise_effect * (dh_dk1_dh_dk2 * E_2 \
                                           + (h_dh_dk1_dtau_dk2 + h_dh_dk2_dtau_dk1) * E_3 \
                                           + hsq_dtau_dk1_dtau_dk2 * E_1)
    return np.linalg.inv(fim)


def signal_generator(current_time, dt_vhc, max_power, signal_freq, delay, measure_dt):
    time = np.arange(current_time - dt_vhc + measure_dt, current_time + measure_dt, measure_dt)

    s = max_power * np.sin((2 * np.pi * signal_freq * (time - delay)) % (2 * np.pi))
    d_s_d_tau = - max_power * 2 * np.pi * signal_freq * np.cos((2 * np.pi * signal_freq * (time - delay)) % (2 * np.pi))

    e_1 = np.sum(np.dot(d_s_d_tau, d_s_d_tau))
    e_2 = np.sum(np.dot(s, s))
    e_3 = np.sum(np.dot(s, d_s_d_tau))

    return e_1, e_2, e_3


def deviation_from_actual_value(array, actual_val):
    return np.sqrt(np.mean(abs(array - actual_val) ** 2, axis=0))

def main():

    data = load_mat('../SimulationData/v2lcRun_sm3_comparisonSoA.mat')
    dp = bound_est_data.params.number_of_skip_data
    # vehicle parameters
    L_1 = data['vehicle']['target']['width']
    L_2 = data['vehicle']['ego']['width']

    rx_area = data['qrx']['f_QRX']['params']['area']

    # time parameters
    time = data['vehicle']['t']['values'][::dp]
    dt = data['vehicle']['t']['dt'] * dp
    start_time = data['vehicle']['t']['start']
    stop_time = data['vehicle']['t']['stop']

    max_power = data['tx']['power']
    signal_freq = bound_est_data.params.signal_freq
    measure_dt = bound_est_data.params.measure_dt

    # relative tgt vehicle positions
    tx1_x = data['vehicle']['target_relative']['tx1_qrx4']['y'][::dp]
    tx1_y = data['vehicle']['target_relative']['tx1_qrx4']['x'][::dp]
    tx2_x = data['vehicle']['target_relative']['tx2_qrx3']['y'][::dp]
    tx2_y = data['vehicle']['target_relative']['tx2_qrx3']['x'][::dp]
    rel_heading = data['vehicle']['target_relative']['heading'][::dp]

    #print("deviation results: ", deviation_from_actual_value(np.asarray(x_becha)[:,:,0], -tx1_x))
    #print(deviation_from_actual_value(np.asarray(x_becha)[:,:,0], -tx1_x).shape)

    # delay parameters
    delay_11 = data['channel']['qrx1']['delay']['tx1'][::dp]
    delay_12 = data['channel']['qrx1']['delay']['tx2'][::dp]
    delay_21 = data['channel']['qrx2']['delay']['tx1'][::dp]
    delay_22 = data['channel']['qrx2']['delay']['tx2'][::dp]

    # received power of QRXes
    pow_qrx1_tx1 = np.array([data['channel']['qrx1']['power']['tx1']['A'][::dp], data['channel']['qrx1']['power']['tx1']['B'][::dp],
                             data['channel']['qrx1']['power']['tx1']['C'][::dp], data['channel']['qrx1']['power']['tx1']['D'][::dp]])
    pow_qrx1_tx2 = np.array([data['channel']['qrx1']['power']['tx2']['A'][::dp], data['channel']['qrx1']['power']['tx2']['B'][::dp],
                             data['channel']['qrx1']['power']['tx2']['C'][::dp], data['channel']['qrx1']['power']['tx2']['D'][::dp]])
    pow_qrx2_tx1 = np.array([data['channel']['qrx2']['power']['tx1']['A'][::dp], data['channel']['qrx2']['power']['tx1']['B'][::dp],
                             data['channel']['qrx2']['power']['tx1']['C'][::dp], data['channel']['qrx2']['power']['tx1']['D'][::dp]])
    pow_qrx2_tx2 = np.array([data['channel']['qrx2']['power']['tx1']['A'][::dp], data['channel']['qrx2']['power']['tx1']['B'][::dp],
                             data['channel']['qrx2']['power']['tx1']['C'][::dp], data['channel']['qrx2']['power']['tx1']['D'][::dp]])

    # noise params
    T = bound_est_data.params.T
    I_bg = bound_est_data.params.I_bg
    p_r_factor = data['qrx']['tia']['shot_P_r_factor']
    i_bg_factor = data['qrx']['tia']['shot_I_bg_factor']
    t_factor1 = data['qrx']['tia']['thermal_factor1']
    t_factor2 = data['qrx']['tia']['thermal_factor1']
    noise_factors = [p_r_factor, i_bg_factor, t_factor1, t_factor2]

    # other params
    rx_fov = bound_est_data.params.rx_fov
    tx_half_angle = bound_est_data.params.tx_half_angle

    # initalize crlb equations with given parameters
    crlb_init_object = CRLB_init(L_1, L_2, rx_area, rx_fov, tx_half_angle)

    # calculate bounds for all elements
    robert_crlb_results = [np.array([]), np.array([])]
    becha_crlb_results = [np.array([]), np.array([]), np.array([]), np.array([])]
    soner_crlb_results = [np.array([]), np.array([]), np.array([]), np.array([])]

    for i in range(len(tx1_x)):
        tx1 = np.array([tx1_x[i], tx1_y[i]])
        tx2 = np.array([tx2_x[i], tx2_y[i]])
        curr_t = time[i]
        delays = np.array([[delay_11[i], delay_12[i]], [delay_21[i], delay_22[i]]])
        powers = np.array([[pow_qrx1_tx1[:, i], pow_qrx1_tx2[:, i]], [pow_qrx2_tx1[:, i], pow_qrx2_tx2[:, i]]])
        fim_inverse_rob = roberts_crlb_single_instance(crlb_init_object, tx1, tx2, delays,
                                     curr_t, dt, max_power, signal_freq, measure_dt, T, I_bg, noise_factors, powers)
        fim_inverse_becha = bechadergue_crlb_single_instance(crlb_init_object, tx1, tx2, delays,
                                     curr_t, dt, max_power, signal_freq, measure_dt, T, I_bg, noise_factors, powers)
        fim_inverse_soner = soner_crlb_single_instance(crlb_init_object, tx1, tx2, delays,
                                     curr_t, dt, max_power, signal_freq, measure_dt, T, I_bg, noise_factors, powers)


        robert_crlb_results[0] = np.append(robert_crlb_results[0], np.sqrt(fim_inverse_rob[0][0]))
        robert_crlb_results[1] = np.append(robert_crlb_results[1], np.sqrt(fim_inverse_rob[1][1]))

        becha_crlb_results[0] = np.append(becha_crlb_results[0], np.sqrt(fim_inverse_becha[0][0]))
        becha_crlb_results[1] = np.append(becha_crlb_results[1], np.sqrt(fim_inverse_becha[1][1]))
        becha_crlb_results[2] = np.append(becha_crlb_results[2], np.sqrt(fim_inverse_becha[2][2]))
        becha_crlb_results[3] = np.append(becha_crlb_results[3], np.sqrt(fim_inverse_becha[3][3]))

        soner_crlb_results[0] = np.append(soner_crlb_results[0], np.sqrt(fim_inverse_soner[0][0]))
        soner_crlb_results[1] = np.append(soner_crlb_results[1], np.sqrt(fim_inverse_soner[1][1]))
        soner_crlb_results[2] = np.append(soner_crlb_results[2], np.sqrt(fim_inverse_soner[2][2]))
        soner_crlb_results[3] = np.append(soner_crlb_results[3], np.sqrt(fim_inverse_soner[3][3]))
        print(i)

    np.savetxt('Data/aoa/crlb_x1.txt', soner_crlb_results[0], delimiter=',')
    np.savetxt('Data/rtof/crlb_x1.txt', becha_crlb_results[0], delimiter=',')
    np.savetxt('Data/aoa/crlb_x2.txt', soner_crlb_results[2], delimiter=',')
    np.savetxt('Data/rtof/crlb_x2.txt', becha_crlb_results[2], delimiter=',')
    np.savetxt('Data/tdoa/crlb_x.txt', robert_crlb_results[0], delimiter=',')

    np.savetxt('Data/aoa/crlb_y1.txt', soner_crlb_results[1], delimiter=',')
    np.savetxt('Data/rtof/crlb_y1.txt', becha_crlb_results[1], delimiter=',')
    np.savetxt('Data/tdoa/crlb_y2.txt', soner_crlb_results[3], delimiter=',')
    np.savetxt('Data/rtof/crlb_y2.txt', becha_crlb_results[3], delimiter=',')
    np.savetxt('Data/tdoa/crlb_y.txt', robert_crlb_results[1], delimiter=',')

    print("finished")
    folder_name = '../GUI_data/means/3/'
    # x_becha, y_becha = np.loadtxt(folder_name+'x_becha_mean.txt', delimiter=','), np.loadtxt(folder_name+'y_becha_mean.txt',
    #                                                                                     delimiter=',')
    # x_roberts, y_roberts = np.loadtxt(folder_name+'x_roberts_mean.txt', delimiter=','), np.loadtxt(folder_name+'y_roberts_mean.txt',
    #                                                                                           delimiter=',')
    # x_soner, y_soner = np.loadtxt(folder_name+'x_pose_mean.txt', delimiter=','), np.loadtxt(folder_name+'y_pose_mean.txt', delimiter=',')




if __name__ == "__main__":
    main()