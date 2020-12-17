from CRLB_init import *
from matfile_read import load_mat
import math
#TODO: correct Soner case for dividing variances


# Roberts' method, CRLB calculation for a single position estimation
def roberts_crlb_single_instance(crlb_obj, tx1, tx2, noise_effect, delays, curr_t, dt_vhc, max_pow, sig_freq, meas_dt):
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

                fim[param1][param2] += noise_effect * (dh_dk1_dh_dk2 * E_2 \
                                       + (h_dh_dk1_dtau_dk2 + h_dh_dk2_dtau_dk1) * E_3 \
                                       + hsq_dtau_dk1_dtau_dk2 * E_1)
    return np.linalg.inv(fim)

#  Bechadergue's method, CRLB calculation for a single position estimation
def bechadergue_crlb_single_instance(crlb_obj, tx1, tx2, noise_effect, delays, curr_t, dt_vhc, max_pow, sig_freq, meas_dt):
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

                fim[param1][param2] += noise_effect * (dh_dk1_dh_dk2 * E_2 \
                                       + (h_dh_dk1_dtau_dk2 + h_dh_dk2_dtau_dk1) * E_3 \
                                       + hsq_dtau_dk1_dtau_dk2 * E_1)
    return np.linalg.inv(fim)

#  Soner's method, CRLB calculation for a single position estimation
def soner_crlb_single_instance(crlb_obj, tx1, tx2, noise_effect, delays, curr_t, dt_vhc, max_pow, sig_freq, meas_dt):
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

                    fim[param1][param2] += noise_effect * (dh_dk1_dh_dk2 * E_2 \
                                           + (h_dh_dk1_dtau_dk2 + h_dh_dk2_dtau_dk1) * E_3 \
                                           + hsq_dtau_dk1_dtau_dk2 * E_1)
    return np.linalg.inv(fim)


def signal_generator(current_time, dt_vhc, max_power, signal_freq, delay, measure_dt):
    time = np.arange(current_time - dt_vhc + measure_dt, current_time + measure_dt, measure_dt)

    e_1, e_2, e_3 = 0, 0, 0

    for t in time:
        s = max_power * math.sin((2 * math.pi * signal_freq * (t - delay)) % (2 * math.pi))
        d_s_d_tau = - max_power * 2 * math.pi * signal_freq * math.cos((2 * math.pi * signal_freq * (t - delay)) % (2 * math.pi))

        e_1 += d_s_d_tau * d_s_d_tau
        e_2 += s * s
        e_3 = s * d_s_d_tau

    return e_1, e_2, e_3


def main():
    data = load_mat('SimulationData/v2lcRun_sm3_comparisonSoA.mat')

    # vehicle parameters
    L_1 = data['vehicle']['target']['width']
    L_2 = data['vehicle']['ego']['width']

    rx_area = data['qrx']['f_QRX']['params']['area']

    # time parameters
    time = data['vehicle']['t']['values']
    dt = data['vehicle']['t']['dt']
    start_time = data['vehicle']['t']['start']
    stop_time = data['vehicle']['t']['stop']

    # TODO: generate signals
    max_power = data['tx']['power']
    signal_freq = 1e6  # 1 MHz signal frequency
    measure_dt = 1 / 2.5e6  # 2.5 MHz measure frequency

    # relative tgt vehicle positions
    tx1_x = data['vehicle']['target_relative']['tx1_qrx4']['y']
    tx1_y = data['vehicle']['target_relative']['tx1_qrx4']['x']
    tx2_x = data['vehicle']['target_relative']['tx2_qrx3']['y']
    tx2_y = data['vehicle']['target_relative']['tx2_qrx3']['x']
    rel_heading = data['vehicle']['target_relative']['heading']

    # delay parameters
    delay_11 = data['channel']['qrx1']['delay']['tx1']
    delay_12 = data['channel']['qrx1']['delay']['tx2']
    delay_21 = data['channel']['qrx2']['delay']['tx1']
    delay_22 = data['channel']['qrx2']['delay']['tx2']

    # noise params
    T = 298  # Kelvin
    p_r_factor = data['qrx']['tia']['shot_P_r_factor']
    i_bg_factor = data['qrx']['tia']['shot_I_bg_factor']
    t_factor1 = data['qrx']['tia']['thermal_factor1']
    t_factor2 = data['qrx']['tia']['thermal_factor1']

    # shot noise var
    var_sq_shot = p_r_factor + i_bg_factor
    var_sq_thermal = T * (t_factor1 + t_factor2)
    var_sq = var_sq_shot + var_sq_thermal

    # other params
    rx_fov = 50  # angle
    tx_half_angle = 60  # angle

    # initalize crlb equations with given parameters
    crlb_init_object = CRLB_init(L_1, L_2, rx_area, rx_fov, tx_half_angle)

    # calculate bounds for all elements
    for i in range(len(tx1_x)):
        tx1 = np.array([tx1_x[i], tx1_y[i]])
        tx2 = np.array([tx2_x[i], tx2_y[i]])
        curr_t = time[i]
        delays = np.array([[delay_11[i], delay_12[i]], [delay_21[i], delay_22[i]]])
        fim_inverse = soner_crlb_single_instance(crlb_init_object, tx1, tx2, 1 / var_sq, delays,
                                     curr_t, dt, max_power, signal_freq, measure_dt)
        print(fim_inverse)
        if i == 100:
            exit(0)


if __name__ == "__main__":
    main()