# In[1]:
from VLC_init import *
from aoa_pos import *
from rtof_pos import *
from Roberts import *
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from math import pi
from scipy.interpolate import interp1d
from mat4py import loadmat
from scipy.io import loadmat, matlab
import math
import sys


# coding: utf-8

## In[1]:
# 'v2lcRun_sm1_laneChange'
# 'v2lcRun_sm2_platoonFormExit'
# 'v2lcRun_sm3_comparisonSoA'
# data_name = 'v2lcRun_sm2_platoonFormExit'
data_name = sys.argv[1]
print(data_name)
data_dir = 'SimulationData/' + data_name + '.mat'
data = loadmat(data_dir)

def load_mat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list)
        else:
            return ndarray

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_vars(data)

def rec_func(data, n):
    for k in data.keys():
        str = ""
        for i in range(n):
            str += "\t"
        print(str, k)
        if isinstance(data[k], dict):
            rec_func(data[k], n + 1)

## In[2]:
data = load_mat(data_dir)
#print(data)

#print(rec_func(data, 0))

# print(data['channel']['qrx1']['power']['tx2']['A'])
#print(type(data['channel']['qrx1']['power']['tx2']['A']))
#print(data['channel']['qrx1']['power']['tx2']['A'].shape)

## In[2]:
vlc_obj = VLC_init()
vlc_obj.distancecar = 1.6
area = data['qrx']['f_QRX']['params']['area']
vlc_obj.rxradius = math.sqrt(area) / math.pi
rel_hdg = data['vehicle']['target_relative']['heading']
vlc_obj.e_angle, vlc_obj.a_angle = 60, 60

tgt_tx1_x = -1 * data['vehicle']['target_relative']['tx1_qrx4']['y']
tgt_tx1_x = tgt_tx1_x[::100]
tgt_tx1_y = data['vehicle']['target_relative']['tx1_qrx4']['x']
tgt_tx1_y = tgt_tx1_y[::100]
tgt_tx2_x = -1 * data['vehicle']['target_relative']['tx2_qrx3']['y']
tgt_tx2_x = tgt_tx2_x[::100]
tgt_tx2_y = data['vehicle']['target_relative']['tx2_qrx3']['x']
tgt_tx2_y = tgt_tx2_y[::100]

ego_qrx1_x = np.zeros(len(tgt_tx1_x))
ego_qrx1_y = np.zeros(len(tgt_tx1_x))
ego_qrx2_x = np.zeros(len(tgt_tx1_x))
ego_qrx2_y = np.zeros(len(tgt_tx1_x)) + vlc_obj.distancecar

## In[2]:
x_pose, y_pose, x_roberts, y_roberts, x_becha, y_becha = [], [], [], [], [], []
x, y = [], []

for i in range(len(tgt_tx1_x)):
    # updating the given coordinates
    print("Iteration #", i, ": ")
    vlc_obj.update_coords(
        ((tgt_tx1_x[i], tgt_tx1_y[i]), (tgt_tx2_x[i], tgt_tx2_y[i])),
        ((ego_qrx1_x[i], ego_qrx1_y[i]), (ego_qrx2_x[i], ego_qrx2_y[i])))
    vlc_obj.update_lookuptable()
    x.append(vlc_obj.trxpos)
    y.append(vlc_obj.trypos)
    vlc_obj.relative_heading = rel_hdg[i]
    # providing the environmentt to methods
    aoa = Pose(vlc_obj)
    rtof = RToF_pos(vlc_obj)
    tdoa = Roberts(vlc_obj)
    # making estimations
    tx_aoa = aoa.estimate()
    print("AoA finished")
    tx_rtof = rtof.estimate()
    print("RToF finished")
    tx_tdoa = tdoa.estimate()
    print("TDoA finished")
    # storing to plot later
    x_pose.append(tx_aoa[0])
    y_pose.append(tx_aoa[1])
    x_becha.append(tx_rtof[0])
    y_becha.append(tx_rtof[1])
    x_roberts.append(tx_tdoa[0])
    y_roberts.append(tx_tdoa[1])
## In[2]:
y_data = np.copy(y)
x_data = np.copy(x)
for i in range(len(y)):
    y_data[i] = y[i][0] + 0.8
    x_data[i] = x[i][0]

plt.figure()
plt.plot(x, y, 'o', color='green', markersize=10)
plt.plot(x_becha, y_becha, 'o', color='orange', markersize=8)
plt.plot(x_pose, y_pose, 'o', color='blue', markersize=7)
plt.plot(x_roberts, y_roberts, 'o', color='purple', markersize=5)
plt.plot(x_data, y_data, '--', color='red', markersize=5)
plt.grid()

green_patch = mpatches.Patch(color='green', label='Actual coordinates')
blue_patch = mpatches.Patch(color='blue', label='AoA-estimated coordinates')
orange_patch = mpatches.Patch(color='orange', label='RToF-estimated coordinates')
purple_patch = mpatches.Patch(color='purple', label='TDoA-estimated coordinates')

plt.legend(handles=[green_patch, blue_patch, orange_patch, purple_patch])
plt.xlabel('x')
plt.ylabel('y')

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise

output_dir = "Figure/"
mkdir_p(output_dir)
name = '{}/' + data_name + '.png'
plt.savefig(name.format(output_dir))

