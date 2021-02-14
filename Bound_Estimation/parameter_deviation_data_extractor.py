import os
import pickle
import numpy as np


def deviation_from_actual_value(array):
    """
    Calculates standard deviation for the parameters
    :param array: either (num_iters, num_points_in_sim, [n] params) or (num_iters, num_points_in_sim, [n*m] params)
    :return:
    """
    if array.ndim == 3:
        deviations = np.zeros((array.shape[1],array.shape[2]))
        for pt in range(array.shape[1]):
            for param in range(array.shape[2]):
                dev = np.std(array[:,pt,param])
                deviations[pt,param] = dev
        return deviations

    elif array.ndim == 4:
        deviations = np.zeros((array.shape[1], array.shape[2], array.shape[3]))
        for pt in range(array.shape[1]):
            for param_ind1 in range(array.shape[2]):
                for param_ind2 in range(array.shape[3]):
                    dev = np.std(array[:, pt, param_ind1, param_ind2])
                    deviations[pt, param_ind1, param_ind2] = dev
        return deviations
    else:
        raise ValueError("Wrong num of dimensions")


def main():
    #retrieving pickle data calculated from parameter_deviation_calculator.py
    directory_path = os.path.dirname(
        os.path.dirname(os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])))  ## directory of directory of file
    pickle_dir = directory_path + '/Bound_Estimation/Parameter_Deviation/'

    with open(pickle_dir + 'theta.pkl', 'rb') as f:
        theta_l_r = pickle.load(f)

    with open(pickle_dir + 'rtof_dist.pkl', 'rb') as f:
        rtof_dist = pickle.load(f)

    with open(pickle_dir + 'tdoa_dist.pkl', 'rb') as f:
        tdoa_dist = pickle.load(f)

    #calculating deviation for theta, rtof_dist.pkl, tdoa_dist
    deviation_theta = deviation_from_actual_value(theta_l_r)
    deviation_rtof_dist = deviation_from_actual_value(rtof_dist)
    deviation_tdoa_dist = deviation_from_actual_value(tdoa_dist)

    #saving calculated deviation parameters.
    with open(pickle_dir + 'deviation_theta.pkl', 'wb') as f:
        pickle.dump(deviation_theta, f)
    with open(pickle_dir + 'deviation_rtof_dist.pkl', 'wb') as f:
        pickle.dump(deviation_rtof_dist, f)
    with open(pickle_dir + 'deviation_tdoa_dist.pkl', 'wb') as f:
        pickle.dump(deviation_tdoa_dist, f)


if __name__ == '__main__':
        main()
