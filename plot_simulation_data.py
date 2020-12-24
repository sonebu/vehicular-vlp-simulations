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

sm = [1, 2, 3]
for i in range(len(sm)):

    if sm[i] == 3:
        input_name = 'v2lcRun_sm3_comparisonSoA'
        fl_name = '/3/'
    elif sm[i] == 2:
        input_name = 'v2lcRun_sm2_platoonFormExit'
        fl_name = '/2/'

    elif sm[i] == 1:
        input_name = 'v2lcRun_sm1_laneChange'
        fl_name = '/1/'

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 20))

    folder_name = 'GUI_data/100_point/' + fl_name
    x, y = np.loadtxt(folder_name+'x.txt', delimiter=','), np.loadtxt(folder_name+'y.txt', delimiter=',')
    x_pose, y_pose = np.loadtxt(folder_name+'x_pose.txt', delimiter=','), np.loadtxt(folder_name+'y_pose.txt', delimiter=',')
    x_becha, y_becha = np.loadtxt(folder_name+'x_becha.txt', delimiter=','), np.loadtxt(folder_name+'y_becha.txt',
                                                                                     delimiter=',')
    x_roberts, y_roberts = np.loadtxt(folder_name+'x_roberts.txt', delimiter=','), np.loadtxt(folder_name+'y_roberts.txt',
                                                                                           delimiter=',')
    x_data, y_data = np.loadtxt(folder_name+'x_data.txt', delimiter=','), np.loadtxt(folder_name+'y_data.txt',
                                                                                  delimiter=',')
    time_ = np.loadtxt(folder_name + 'time.txt', delimiter=',')
    rel_hdg = np.loadtxt(folder_name + 'rel_hdg.txt', delimiter=',')


    #f.suptitle('Fig 1: Relative Target Vehicle Trajectory \n Fig 2: x Estimation Results \n Fig 3: y Estimation Results')
    # img_ego = ndimage.rotate(plt.imread('red_racing_car_top_view_preview.png'), 0)
    img_tgt_s = ndimage.rotate(plt.imread('green_racing_car_top_view_preview.png'), rel_hdg[0])
    img_tgt_f = ndimage.rotate(plt.imread('green_racing_car_top_view_preview.png'), rel_hdg[-1])
    if fl_name == '/3/':
        ax1.add_artist(
            AnnotationBbox(OffsetImage(plt.imread('red_racing_car_top_view_preview.png'), zoom=0.25), (0.2, -0.12),
                           frameon=False))

        ax1.add_artist(AnnotationBbox(OffsetImage(plt.imread('green_racing_car_top_view_preview.png'), zoom=0.08),
                                      (x[0][0] - 0.27, y[0][0] + 0.2), frameon=False))
        ax1.add_artist(AnnotationBbox(OffsetImage(plt.imread('green_racing_car_top_view_preview.png'), zoom=0.08),
                                      (x[-1][0] - 0.27, y[-1][0] + 0.2), frameon=False))
    else:
        ax1.add_artist(
            AnnotationBbox(OffsetImage(plt.imread('red_racing_car_top_view_preview.png'), zoom=0.25), (0, 0),
                           frameon=False))

        ax1.add_artist(AnnotationBbox(OffsetImage(img_tgt_s, zoom=0.08), (x[0][0], y[0][0]), frameon=False))
        ax1.add_artist(AnnotationBbox(OffsetImage(img_tgt_f, zoom=0.08), (x[-1][0], y[-1][0]), frameon=False))


    # ax1.add_artist(AnnotationBbox(OffsetImage(img_tgt_f, zoom=0.05), (x_data[-1][0], y_data[-1][0]), frameon=False))
    if fl_name == '/2/':
        ax1.plot(x[:, 0], y[:, 0], 'o', color='green', markersize=10)
        ax1.title.set_text('Fig.1: Relative Target Vehicle Trajectory')
        ax1.plot(x[:, 0], y[:, 0], '-', color='red', markersize=5)
    else:
        ax1.plot(x[:, 0], y[:, 0], 'o', color='green', markersize=10)
        ax1.title.set_text('Fig.1: Relative Target Vehicle Trajectory')
        ax1.plot(x[:, 0], y[:, 0], '-', color='red', markersize=5)
        mid = 4
        arrow_x = x[mid, 0]
        arrow_y = y[mid, 0]
        if fl_name == '/3/':
            ax1.arrow(arrow_x, arrow_y, -0.5, 0, width=0.05)
    if fl_name == '/3/':
        ax1.set_xlim(-8, 1)
        ax1.set_ylim(-1, 4)
    elif fl_name == '/2/':
        ax1.set_xlim(-10, 1)
        ax1.set_ylim(-5, 5)
    elif fl_name == '/1/':
        ax1.set_xlim(-9, 1)
        ax1.set_ylim(-3, 3)
    ax1.grid()
    green_patch = mpatches.Patch(color='green', label='Target Vehicle')
    red_patch = mpatches.Patch(color='red', label='Ego Vehicle')
    ax1.legend(handles=[green_patch, red_patch])
    ax1.set_xlabel('Ego Frame x [m]')
    ax1.set_ylabel('Ego Frame y [m]')

    ax2.plot(time_, x[:, 0], 'o', color='green')

    ax2.plot(time_, x_pose[:, 0], '-', color='blue')
    ax2.plot(time_, x_roberts[:, 0], '-', color='purple')
    ax2.plot(time_, x_becha[:, 0], '-', color='orange')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('[m]')
    if fl_name == '/3/':
        ax2.set_ylim(x[0,0]-0.5,x[-1,0]+0.5)
        ax2.set_xlim(0, 9)
    elif fl_name == '/1/':
        ax2.set_xlim(0,1)
        ax2.set_ylim(-7, -2)
    elif fl_name == '/2/':
        ax2.set_xlim(0,1)
        ax2.set_ylim(-9, -2)
    ax2.grid()
    green_patch = mpatches.Patch(color='green', label='Actual coordinates')
    blue_patch = mpatches.Patch(color='blue', label='AoA-estimated coordinates')
    orange_patch = mpatches.Patch(color='orange', label='RToF-estimated coordinates')
    purple_patch = mpatches.Patch(color='purple', label='TDoA-estimated coordinates')
    ax2.legend(handles=[green_patch, blue_patch, orange_patch, purple_patch])
    ax2.title.set_text('Fig 2: x Estimation Results')
    ax3.title.set_text('Fig 3: y Estimation Results')
    ax3.plot(time_[i], y[i, 0], 'o', color='green')

    ax3.plot(time_, y_pose[:, 0], '-', color='blue')
    ax3.plot(time_, y_roberts[:, 0], '-', color='purple')
    ax3.plot(time_, y_becha[:, 0], '-', color='orange')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('[m]')
    if fl_name == '/3/':
        ax3.set_ylim(y[0,0]-1,y[-1,0]+1)
        ax3.set_xlim(0, 9)
    elif fl_name == '/1/':
        ax3.set_xlim(0, 1)
        ax3.set_ylim(-4,4)
    elif fl_name == '/2/':
        ax3.set_xlim(0,1)
        ax3.set_ylim(-5, 5)
    ax3.grid()
    green_patch = mpatches.Patch(color='green', label='Actual coordinates')
    blue_patch = mpatches.Patch(color='blue', label='AoA-estimated coordinates')
    orange_patch = mpatches.Patch(color='orange', label='RToF-estimated coordinates')
    purple_patch = mpatches.Patch(color='purple', label='TDoA-estimated coordinates')
    ax3.legend(handles=[green_patch, blue_patch, orange_patch, purple_patch])

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
    name = '{}/' + input_name + '.png'
    plt.savefig(name.format(output_dir))