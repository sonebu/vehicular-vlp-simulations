import os
import pickle
import numpy as np
import math
from Bound_Estimation.matfile_read import *
from config import gen_sim_data

def deviation_from_actual_value(array, actual_val):
    return np.sqrt(np.mean(np.abs(array - actual_val) ** 2, axis=0))

def main():
    directory_path = os.path.dirname(
        os.path.dirname(os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])))  ## directory of directory of file
    data_name = gen_sim_data.names.data_names[2]
    data_dir = directory_path + '/SimulationData/' + data_name + '.mat'
    data = load_mat(data_dir)
    L = data['vehicle']['target']['width']
    """getting actual coordinates from .mat file"""
    tx1_x = (-1)*data['vehicle']['target_relative']['tx1_qrx4']['y'][::gen_sim_data.params.number_of_skip_data]
    tx1_y = data['vehicle']['target_relative']['tx1_qrx4']['x'][::gen_sim_data.params.number_of_skip_data]
    tx2_x = (-1)*data['vehicle']['target_relative']['tx2_qrx3']['y'][::gen_sim_data.params.number_of_skip_data]
    tx2_y = data['vehicle']['target_relative']['tx2_qrx3']['x'][::gen_sim_data.params.number_of_skip_data]

    """calculating actual theta"""
    aoa11 = np.arctan(tx1_y/tx1_x)
    aoa12 = np.arctan(tx2_y/tx2_x)
    aoa21 = np.arctan((tx1_y - L) / tx1_x)
    aoa22 = np.arctan((tx2_y - L) / tx2_x)
    thetas = np.array([[aoa11, aoa12], [aoa21, aoa22]]).transpose()
    """calculating actual distance values"""
    d11 = np.sqrt(np.power(tx1_x, 2) + np.power(tx1_y, 2))
    d12 = np.sqrt(np.power(tx2_x, 2) + np.power(tx2_y, 2))
    d21 = np.sqrt(np.power(tx1_x, 2) + np.power((tx1_y-L), 2))
    d22 = np.sqrt(np.power(tx1_x, 2) + np.power((tx1_y-L), 2))
    rtof_distances = np.array([[d11, d12], [d21, d22]]).transpose()
    dA = d11 - d21
    dB = d12 - d22
    tdoa_distances = np.array([dA, dB], np.newaxis).transpose()

    """retrieving pickle data calculated from parameter_deviation_calculator.py"""
    folder_name = gen_sim_data.names.folder_names[2]
    data_point = str(int(1000 / gen_sim_data.params.number_of_skip_data)) + '_point_' + '/'
    f_name = directory_path + '/Parameter_Deviation/' + data_point + folder_name
    if not os.path.exists(f_name):
        os.makedirs(f_name)
    directory_path = os.path.dirname(
        os.path.dirname(os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])))  ## directory of directory of file
    pickle_dir = directory_path + '/Bound_Estimation/Parameter_Deviation/'
    with open(pickle_dir + 'theta.pkl', 'rb') as f:
        theta_l_r = pickle.load(f)
        print(theta_l_r.shape)
    """""
    with open(pickle_dir + 'rtof_dist.pkl', 'rb') as f:
        rtof_dist = pickle.load(f)
        print(rtof_dist.shape)
    """
    with open(pickle_dir + 'tdoa_dist.pkl', 'rb') as f:
        tdoa_dist = pickle.load(f)
        print(tdoa_dist.shape)

    """calculating deviation for theta, rtof_dist, tdoa_dist"""
    deviation_theta = np.copy(theta_l_r)
    for i in range(len(deviation_theta)):
        deviation_theta[i] = deviation_from_actual_value(theta_l_r[i], thetas)
    """
    deviation_rtof_dist = np.copy(rtof_dist)
    for i in range(len(deviation_rtof_dist)):
        deviation_rtof_dist[i] = deviation_from_actual_value(rtof_dist[i], rtof_distances)
    """
    deviation_tdoa_dist = np.copy(tdoa_dist)
    for i in range(len(deviation_tdoa_dist)):
        deviation_tdoa_dist[i] = deviation_from_actual_value(tdoa_dist[i], tdoa_distances)

    """saving calculated deviation parameters."""
    with open(pickle_dir + 'deviation_theta.pkl', 'wb') as f:
        pickle.dump(deviation_theta, f)
    #with open(pickle_dir + 'deviation_rtof_dist', 'wb') as f:
    #    pickle.dump(deviation_rtof_dist, f)
    with open(pickle_dir + 'deviation_tdoa_dist.pkl', 'wb') as f:
        pickle.dump(deviation_tdoa_dist, f)


if __name__ == '__main__':
        main()
