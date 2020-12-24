# In[1]:
from VLP_methods.VLC_init import *
from VLP_methods.aoa import *
from VLP_methods.rtof import *
from VLP_methods.tdoa import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import pi
from scipy.interpolate import interp1d
from mat4py import loadmat
from scipy.io import loadmat, matlab
import math
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import sys
import cProfile
from scipy import ndimage


# coding: utf-8

## In[1]:
# 'v2lcRun_sm1_laneChange'
# 'v2lcRun_sm2_platoonFormExit'
# 'v2lcRun_sm3_comparisonSoA'

data_name = 'v2lcRun_sm3_comparisonSoA'
# data_name = sys.argv[1]
print(data_name)
data_dir = 'SimulationData/' + data_name + '.mat'
data = loadmat(data_dir)
folder_name = '3/'
dp = 100
data_point = '100_point/'
import cProfile, pstats
profiler = cProfile.Profile()
profiler.enable()



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
# print(data)

# print(rec_func(data, 0))

# print(data['channel']['qrx1']['power']['tx2']['A'])
# print(type(data['channel']['qrx1']['power']['tx2']['A']))
# print(data['channel']['qrx1']['power']['tx2']['A'].shape)

## In[2]:
vlc_obj = VLC_init()
vlc_obj.distancecar = 1.6
area = data['qrx']['f_QRX']['params']['area']
vlc_obj.rxradius = math.sqrt(area) / math.pi
rel_hdg = data['vehicle']['target_relative']['heading']
rel_hdg = rel_hdg[::dp]
vlc_obj.e_angle, vlc_obj.a_angle = 60, 60
time_ = data['vehicle']['t']['values']
time_ = time_[::dp]

tgt_tx1_x = -1 * data['vehicle']['target_relative']['tx1_qrx4']['y']
tgt_tx1_x = tgt_tx1_x[::dp]
tgt_tx1_y = data['vehicle']['target_relative']['tx1_qrx4']['x']
tgt_tx1_y = tgt_tx1_y[::dp]
tgt_tx2_x = -1 * data['vehicle']['target_relative']['tx2_qrx3']['y']
tgt_tx2_x = tgt_tx2_x[::dp]
tgt_tx2_y = data['vehicle']['target_relative']['tx2_qrx3']['x']
tgt_tx2_y = tgt_tx2_y[::dp]

ego_qrx1_x = np.zeros(len(tgt_tx1_x))
ego_qrx1_y = np.zeros(len(tgt_tx1_x)) - vlc_obj.distancecar /2
ego_qrx2_x = np.zeros(len(tgt_tx1_x))
ego_qrx2_y = np.zeros(len(tgt_tx1_x)) + vlc_obj.distancecar /2

## In[2]:
x, y, x_pose, y_pose, x_roberts, y_roberts, x_becha, y_becha = np.zeros((len(tgt_tx1_x), 2)), np.zeros((len(tgt_tx1_x),
                                                                                                        2)), \
                                                               np.zeros((len(tgt_tx1_x), 2)), np.zeros((len(tgt_tx1_x),
                                                                                                        2)), \
                                                               np.zeros((len(tgt_tx1_x), 2)), np.zeros((len(tgt_tx1_x),
                                                                                                        2)), \
                                                               np.zeros((len(tgt_tx1_x), 2)), np.zeros((len(tgt_tx1_x),
                                                                                                        2))

for i in range(len(tgt_tx1_x)):
    # updating the given coordinates
    print("Iteration #", i, ": ")
    vlc_obj.update_coords(
        ((tgt_tx1_x[i], tgt_tx1_y[i]), (tgt_tx2_x[i], tgt_tx2_y[i])),
        ((ego_qrx1_x[i], ego_qrx1_y[i]), (ego_qrx2_x[i], ego_qrx2_y[i])))
    vlc_obj.update_lookuptable()
    x[i] = vlc_obj.trxpos
    y[i] = vlc_obj.trypos
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
    x_pose[i] = tx_aoa[0]
    y_pose[i] = tx_aoa[1]
    x_becha[i] = tx_rtof[0]
    y_becha[i] = tx_rtof[1]
    x_roberts[i] = tx_tdoa[0]
    y_roberts[i] = tx_tdoa[1]
## In[2]:
y_data = np.copy(y)
x_data = np.copy(x)
for i in range(len(y)):
    y_data[i] = (y[i][0] + y[i][1])/2
    x_data[i] = (x[i][0] + x[i][1])/2

np.savetxt('GUI_data/'+data_point+folder_name+'/x.txt', x, delimiter=',')
np.savetxt('GUI_data/'+data_point+folder_name+'/x_pose.txt', x_pose, delimiter=',')
np.savetxt('GUI_data/'+data_point+folder_name+'/x_becha.txt', x_becha, delimiter=',')
np.savetxt('GUI_data/'+data_point+folder_name+'/x_roberts.txt', x_roberts, delimiter=',')
np.savetxt('GUI_data/'+data_point+folder_name+'/x_data.txt', x_data, delimiter=',')
np.savetxt('GUI_data/'+data_point+folder_name+'/y.txt', y, delimiter=',')
np.savetxt('GUI_data/'+data_point+folder_name+'/y_data.txt', y_data, delimiter=',')
np.savetxt('GUI_data/'+data_point+folder_name+'/y_becha.txt', y_becha, delimiter=',')
np.savetxt('GUI_data/'+data_point+folder_name+'/y_roberts.txt', y_roberts, delimiter=',')
np.savetxt('GUI_data/'+data_point+folder_name+'/y_pose.txt', y_pose, delimiter=',')
np.savetxt('GUI_data/'+data_point+folder_name+'/time.txt', time_, delimiter=',')
np.savetxt('GUI_data/'+data_point+folder_name+'/rel_hdg.txt', rel_hdg, delimiter=',')

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('ncalls')
# stats.print_stats()


#axs[0].plot(t1, f(t1), 'o', t2, f(t2), '-')