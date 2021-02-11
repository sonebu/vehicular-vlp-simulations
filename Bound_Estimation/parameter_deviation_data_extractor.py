import os
import pickle

import numpy as np

from config import gen_sim_data

def deviation_from_actual_value(array, actual_val):
    return np.sqrt(np.mean(abs(array - actual_val) ** 2, axis=0))

def main():
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
