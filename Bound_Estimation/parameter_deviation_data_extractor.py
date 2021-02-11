import os
import pickle
import numpy as np
from Bound_Estimation.matfile_read import *
from config import gen_sim_data

def deviation_from_actual_value(array, actual_val):
    return np.sqrt(np.mean(abs(array - actual_val) ** 2, axis=0))

def main():
    directory_path = os.path.dirname(
        os.path.dirname(os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])))  ## directory of directory of file
    data_name = gen_sim_data.names.data_names[2]
    data_dir = directory_path + '/SimulationData/' + data_name + '.mat'
    data = load_mat(data_dir)

    tx1_x = data['vehicle']['target_relative']['tx1_qrx4']['y'][::gen_sim_data.params.number_of_skip_data]
    tx1_y = data['vehicle']['target_relative']['tx1_qrx4']['x'][::gen_sim_data.params.number_of_skip_data]
    tx2_x = data['vehicle']['target_relative']['tx2_qrx3']['y'][::gen_sim_data.params.number_of_skip_data]
    tx2_y = data['vehicle']['target_relative']['tx2_qrx3']['x'][::gen_sim_data.params.number_of_skip_data]

    folder_name = gen_sim_data.names.folder_names[2]
    dp = gen_sim_data.params.number_of_skip_data

    data_point = str(int(1000 / dp)) + '_point_' + '/'

    f_name = directory_path + '/Parameter_Deviation/' + data_point + folder_name

    if not os.path.exists(f_name):
        os.makedirs(f_name)

    max_power = data['tx']['power']
    directory_path = os.path.dirname(
        os.path.dirname(os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])))  ## directory of directory of file

    pickle_dir = directory_path + '/Bound_Estimation/Parameter_Deviation/'
    with open(pickle_dir + 'theta.pkl', 'rb') as f:
        theta_l_r = pickle.load(f)
        print(theta_l_r.shape)
    with open(pickle_dir + 'tdoa_dist.pkl', 'rb') as f:
        tdoa_dist = pickle.load(f)
        print(tdoa_dist.shape)


if __name__ == '__main__':
        main()
