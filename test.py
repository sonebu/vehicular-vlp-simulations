import numpy as np
import time
import os
import subprocess
import multiprocessing
import resource
import matplotlib.pyplot as plt
import math
from functools import lru_cache
import scipy
from scipy.integrate import quad, dblquad, tplquad
from pose_estimation import *
from VLC_init import *



vlc_obj = VLC_init()
tx_cord_1, tx_cord_2, tx_cord_3, tx_cord_4 = np.array([[-5, 2], [-5, 3]]), np.array([[-5, 2], [-5, 3]]), np.array \
    ([[-5, 2], [-5, 3]]), np.array([[-5, 2], [-5, 3]])
rx_cord_1, rx_cord_2, rx_cord_3, rx_cord_4 = np.array([[-5, 2], [-5, 3]]), np.array([[-5, 2], [-5, 3]]), np.array(
    [[-5, 2], [-5, 3]]), np.array([[-5, 2], [-5, 3]])
tx_cords = [tx_cord_1, tx_cord_2, tx_cord_3, tx_cord_4]
rx_cords = [rx_cord_1, rx_cord_2, rx_cord_3, rx_cord_4]
x_pose, y_pose, x_roberts, y_roberts, x_becha, y_becha = [], [], [], [], [], []
for i in range(len(tx_cords)):
    vlc_obj.update_cords(tx_cords[i], rx_cords[i])
    vlc_obj.update_lookuptable()
    pose_estimation = pose_estimation(vlc_obj)
    tx_pose = pose_estimation.estimate()
    roberts_estimation = roberts(vlc_obj)
    tx_roberts = roberts_estimation.estimate()
    becha_estimation = becha(vlc_obj)
    tx_becha = becha_estimation.estimate()
    x_pose.append([tx_pose[0][0], tx[1][0]])
    y_pose.append([tx_[0][1], tx[1][1]])
    x_becha.append([tx[0][0], tx[1][0]])
    y_becha.append([tx[0][1], tx[1][1]])
    x_roberts.append([tx[0][0], tx[1][0]])
    y_roberts.append([tx[0][1], tx[1][1]])

plt.figure()
plt.plot(x_pose, y_pose, 'o', color='blue')
plt.plot(x_becha, y_becha, 'o', color='orange')
plt.plot(x_roberts, y_roberts, 'o', color='purple')
plt.plot(vlc_obj.trxpos, vlc_obj.trypos, 'o', color='green')
plt.xlim(-6, -4)
plt.ylim(1, 4)
plt.grid()


