from CRLB_init import *
from matfile_read import load_mat

#TODO: add sigma square values to the calculations


def roberts_crlb_single_instance(crlb_obj, tx1, tx2, E_1, E_2, E_3, noise_effect):
    flag = False
    fim = np.zeros(shape=(2,2))

    for param1, param2 in zip(range(2), range(2)):
        for i in range(2):
            for j in range(2):
                ij = (i + 1)*10 + (j + 1)
                h_ij = crlb_obj.get_h_ij(ij, flag)
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
    return fim


def bechadergue_crlb_single_instance(crlb_obj, tx1, tx2, E_1, E_2, E_3, noise_effect):
    fim = np.zeros(shape=(4, 4))

    for param1, param2 in zip(range(4), range(4)):
        for i in range(2):
            for j in range(2):
                ij = (i + 1) * 10 + (j + 1)
                h_ij = crlb_obj.get_h_ij(ij)
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
    return fim


def soner_crlb_single_instance(crlb_obj, tx1, tx2, E_1, E_2, E_3, noise_effect):
    fim = np.zeros(shape=(4, 4))

    for param1, param2 in zip(range(4), range(4)):
        for i in range(2):
            for j in range(2):
                for qrx in range(4):
                    ij = (i + 1) * 10 + (j + 1)
                    q = qrx + 1

                    h_ijq = crlb_obj.get_h_ijq(ij, q)
                    dh_dk1 = crlb_obj.get_d_hij_q_d_param(param1 + 1, ij, q)
                    dh_dk2 = crlb_obj.get_d_hij_q_d_param(param2 + 1, ij, q)
                    dtau_dk1 = crlb_obj.get_d_tau_d_param(param1 + 1, ij, tx1, tx2)
                    dtau_dk2 = crlb_obj.get_d_tau_d_param(param2 + 1, ij, tx1, tx2)

                    dh_dk1_dh_dk2 = dh_dk1 * dh_dk2
                    h_dh_dk1_dtau_dk2 = - h_ijq * dh_dk1 * dtau_dk2
                    h_dh_dk2_dtau_dk1 = - h_ijq * dh_dk2 * dtau_dk1
                    hsq_dtau_dk1_dtau_dk2 = h_ijq ** 2 * dtau_dk1 * dtau_dk2

                    fim[param1][param2] += noise_effect * (dh_dk1_dh_dk2 * E_2 \
                                           + (h_dh_dk1_dtau_dk2 + h_dh_dk2_dtau_dk1) * E_3 \
                                           + hsq_dtau_dk1_dtau_dk2 * E_1)
    return fim


def main():
    data = load_mat('SimulationData/v2lcRun_sm3_comparisonSoA.mat')

    # vehicle parameters
    L_1 = data['vehicle']['target']['width']
    L_2 = data['vehicle']['ego']['width']

    rx_area = data['qrx']['f_QRX']['params']['area']

    # time parameters
    time = data['vehicle']['t']['values']
    dt = data['vehicle']['t']['dt']

    # TODO: generate signals
    max_power = data['tx']['power']

    # relative tgt vehicle positions
    tx1_x = data['vehicle']['target_relative']['tx1_qrx4']['y']
    tx1_y = data['vehicle']['target_relative']['tx1_qrx4']['x']
    tx2_x = data['vehicle']['target_relative']['tx2_qrx3']['y']
    tx2_y = data['vehicle']['target_relative']['tx2_qrx3']['x']
    rel_heading = data['vehicle']['target_relative']['heading']

    # noise params
    k = data['qrx']['tia']['params']['boltzmann']
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




if __name__ == "__main__":
    main()