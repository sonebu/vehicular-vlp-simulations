import matplotlib.pyplot as plt
from VLC_init import *
from aoa_pos import *
from rtof_pos import *
from Roberts import *
import numpy as np
from matplotlib import pyplot as plt
from math import pi
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

vlc_obj = VLC_init()

u=0.5     #x-position of the center
v=-5.    #y-position of the center
a=5.     #radius on the x-axis
b=5.5    #radius on the y-axis

t = np.linspace(pi, 2*pi, 100)
x_data = u+a*np.cos(t)
y_data = v+b*np.sin(t)

# Interpolate values for x and y.
t2 = np.linspace(pi, 2*pi, 8)
# One-dimensional linear interpolation.
y2 = np.interp(t2, t, x_data)
x2 = np.interp(t2, t, y_data)

tx_cords = []
for i in range(len(x2)):
    tx_cords.append(((x2[i], y2[i]),(x2[i], y2[i] + 1)))

rx_cord = ((0, 0), (0, 1))

x_pose, y_pose, x_roberts, y_roberts, x_becha, y_becha = [], [], [], [], [], []
x, y = [], []
for i in range(len(tx_cords)):
    # updating the given coordinates

    print("Iteration #", i ,": ")
    vlc_obj.update_coords(tx_cords[i], rx_cord)
    vlc_obj.update_lookuptable()
    x.append(vlc_obj.trxpos)
    y.append(vlc_obj.trypos)
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

fig, ax = plt.subplots()
ax.set_xlim(-11, 1)
ax.set_ylim(-5, 7)

img = mpimg.imread('red_racing_car_top_view_preview.png')
imagebox = OffsetImage(img, zoom=0.3)
ab = AnnotationBbox(imagebox, (0, 0))

ax.add_artist(ab)
fig.tight_layout()
plt.draw()
#plt.figure()
plt.plot(x, y,'o', color='green', facecolor=None, markersize=14)
plt.plot(x_becha, y_becha, 'o', color='blue', facecolor=None, markersize=12)
plt.plot(x_pose, y_pose, 'o', color='orange',facecolor=None, markersize=11)
plt.plot(x_roberts, y_roberts, 'o', color='purple', facecolor=None, markersize=9)
plt.plot(y_data, x_data+0.5, '--', color='red', markersize=9)
plt.grid()

green_patch = mpatches.Patch(color='green', label='Actual coordinates')
blue_patch = mpatches.Patch(color='orange', label='AoA-estimated coordinates')
orange_patch = mpatches.Patch(color='blue', label='RToF-estimated coordinates')
purple_patch = mpatches.Patch(color='purple', label='TDoA-estimated coordinates')

plt.legend(handles=[green_patch, blue_patch, orange_patch, purple_patch])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Estimation Results for Different Methods')
plt.show()