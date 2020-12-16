from CRLB_init import *
from matfile_read import load_mat

#TODO: add sigma square values to the calculations

def roberts_crlb_single_instance(crlb_init_object, tx1, tx2):
    flag = False
    FIM = np.zeros(shape=(2,2))

    for param1, param2 in zip(range(2), range(2)):
        for i in range(2):
            for j in range(2):
                ij = (i + 1)*10 + (j + 1)





def bechadergue_crlb_single_instance(crlb_init_object, tx1, tx2):


def soner_crlb_single_instance(crlb_init_object, tx1, tx2):



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

    #other params
    rx_fov = 50  # angle
    tx_half_angle = 60  # angle

    crlb_init_object = CRLB_init(L_1, L_2, rx_area, rx_fov, tx_half_angle)


if __name__ == "__main__":
    main()