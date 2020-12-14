from scipy.io import loadmat
import numpy as np
import h5py

# from mat4py import loadmat

data_name = 'v2lcRun_sm1_laneChange.mat'
data = loadmat(data_name)
tgt_name = 'intsc_tgt.mat'
ego_name = 'intsc_ego.mat'
time_name = 'intsc_time.mat'
tgt = loadmat(tgt_name)
ego = loadmat(tgt_name)
time = loadmat(time_name)
rel_hdg = tgt['hdg']
tgt_left_tail_x = tgt['left_tail_x']
tgt_left_tail_y = tgt['left_tail_y']
tgt_right_tail_x = tgt['right_tail_x']
tgt_right_tail_y = tgt['right_tail_y']
ego_x = ego['x'].T
ego_y = ego['y'].T