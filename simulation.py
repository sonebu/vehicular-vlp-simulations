import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.figure import SubplotParams
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import tkinter as t
import tkinter as tk
from scipy import ndimage
from tkinter import ttk
from tkinter import *
import matplotlib.patches as mpatches
from subprocess import call
from functools import partial
from PIL import ImageTk, Image
import time
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

matplotlib.use("TkAgg")
LARGE_FONT = ("Verdana", 12)
style.use("ggplot")
global f, ax1, ax2, ax3
f = Figure(figsize=(40, 40), subplotpars=SubplotParams(hspace=0.5))
ax1 = f.add_subplot(311)
ax2 = f.add_subplot(312)
ax3 = f.add_subplot(313)

global img
global input_name
global name

input_name = 'v2lcRun_sm3_comparisonSoA'
global res
res = FALSE

def start_page(controller):
    global res
    res = TRUE
    controller.show_frame(StartPage)




def run(sm):
    # first key text
    global fl_name, x, y, x_pose, y_pose, x_becha, y_becha, x_roberts, y_roberts,x_data, y_data, time_, rel_hdg
    global ani
    if sm == 3:
        input_name = 'v2lcRun_sm3_comparisonSoA'
        fl_name = '/3/'
    elif sm == 2:
        input_name = 'v2lcRun_sm2_platoonFormExit'
        fl_name = '/2/'

    elif sm == 1:
        input_name = 'v2lcRun_sm1_laneChange'
        fl_name = '/1/'

    folder_name = 'GUI_data/1000_point/' + fl_name
    x, y = np.loadtxt(folder_name + 'x.txt', delimiter=','), np.loadtxt(folder_name + 'y.txt', delimiter=',')
    x_pose, y_pose = np.loadtxt(folder_name + 'x_pose.txt', delimiter=','), np.loadtxt(folder_name + 'y_pose.txt',
                                                                                       delimiter=',')
    x_becha, y_becha = np.loadtxt(folder_name + 'x_becha.txt', delimiter=','), np.loadtxt(folder_name + 'y_becha.txt',
                                                                                          delimiter=',')
    x_roberts, y_roberts = np.loadtxt(folder_name + 'x_roberts.txt', delimiter=','), np.loadtxt(
        folder_name + 'y_roberts.txt',
        delimiter=',')
    x_data, y_data = np.loadtxt(folder_name + 'x_data.txt', delimiter=','), np.loadtxt(folder_name + 'y_data.txt',
                                                                                       delimiter=',')
    time_ = np.loadtxt(folder_name + 'time.txt', delimiter=',')
    rel_hdg = np.loadtxt(folder_name + 'rel_hdg.txt', delimiter=',')
    x, y = x[::20], y[::20]
    x_pose, y_pose = x_pose[::20], y_pose[::20]
    x_roberts, y_roberts = x_roberts[::20], y_roberts[::20]
    x_becha, y_becha = x_becha[::20], y_becha[::20]
    x_data, y_data = x_data[::20], y_data[::20]
    time_, rel_hdg = time_[::20], rel_hdg[::20]

    app = SeaofBTCapp()
    ani = animation.FuncAnimation(f, animate, repeat_delay=1)
    #ani.save('sim.mp4', writer='ffmpeg', fps=30)

    app.mainloop()



def animate(i):
    global res, f, ax1, ax2, ax3, fl_name, fl_name, x, y, x_pose, y_pose, x_becha, y_becha, x_roberts, y_roberts, x_data, y_data, time_, rel_hdg
    if res == TRUE:
        f.clf()
        ax1 = f.add_subplot(311)
        ax2 = f.add_subplot(312)
        ax3 = f.add_subplot(313)

        res = FALSE


    #f.suptitle('Fig 1: Relative Target Vehicle Trajectory \n Fig 2: x Estimation Results \n Fig 3: y Estimation Results')
    img_ego_s = ndimage.rotate(plt.imread('red_racing_car_top_view_preview.png'), rel_hdg[0])

    img_tgt_s = ndimage.rotate(plt.imread('green_racing_car_top_view_preview.png'), rel_hdg[0])
    img_tgt_f = ndimage.rotate(plt.imread('green_racing_car_top_view_preview.png'), rel_hdg[-1])
    if fl_name == '/3/':
        ax1.add_artist(
            AnnotationBbox(OffsetImage(img_ego_s, zoom=0.25), (0.2, -0.12),
                           frameon=False))

        ax1.add_artist(AnnotationBbox(OffsetImage(plt.imread('green_racing_car_top_view_preview.png'), zoom=0.08),
                                      (x[0][0] - 0.27, y[0][0] + 0.2), frameon=False))
        if i == (len(x_data) -1):
            ax1.add_artist(AnnotationBbox(OffsetImage(img_tgt_f, zoom=0.08),
                                      (x[-1][0]- 0.27, y[-1][0] + 0.2), frameon=False))

    elif fl_name == '/2/':
        ax1.add_artist(
            AnnotationBbox(OffsetImage(plt.imread('red_racing_car_top_view_preview.png'), zoom=0.25), (0, 0),
                           frameon=False))

        ax1.add_artist(AnnotationBbox(OffsetImage(img_tgt_s, zoom=0.08),
                                      (x[1][0], y[1][0]), frameon=False))
        if i == (len(x_data) - 1):
            ax1.add_artist(AnnotationBbox(OffsetImage(img_tgt_f, zoom=0.08),
                                      (x[-1][0], y[-1][0]), frameon=False))

    elif fl_name == '/1/':
        ax1.add_artist(
            AnnotationBbox(OffsetImage(plt.imread('red_racing_car_top_view_preview.png'), zoom=0.25), (0, 0),
                           frameon=False))

        ax1.add_artist(AnnotationBbox(OffsetImage(img_tgt_s, zoom=0.08),
                                      (x[0][0], y[0][0]), frameon=False))
        if i == (len(x_data) - 1):
            ax1.add_artist(AnnotationBbox(OffsetImage(img_tgt_f, zoom=0.08),
                                      (x[-1][0], y[-1][0]), frameon=False))


    # ax1.add_artist(AnnotationBbox(OffsetImage(img_tgt_f, zoom=0.05), (x_data[-1][0], y_data[-1][0]), frameon=False))
    if fl_name == '/2/':
        if i > 0:
            ax1.plot(x[i, 0], y[i, 0], 'o', color='green', markersize=10)
            ax1.title.set_text('Fig.1: Relative Target Vehicle Trajectory')
            ax1.plot(x[i, 0], y[i, 0], '-', color='red', markersize=5)

    else:
        ax1.plot(x[i, 0], y[i, 0], 'o', color='green', markersize=10)
        ax1.title.set_text('Fig.1: Relative Target Vehicle Trajectory')
        ax1.plot(x[i, 0], y[i, 0], '-', color='red', markersize=5)
        mid = 25
        arrow_x = x[mid, 0]
        arrow_y = y[mid, 0]
        if i == mid and fl_name == '/3/':
            ax1.arrow(arrow_x, arrow_y, -0.5, 0, width=0.05)
    if fl_name == '/3/':
        ax1.set_xlim(-8, 1)
        ax1.set_ylim(-1, 4)
    elif fl_name == '/2/':
        ax1.set_xlim(-10, 1)
        ax1.set_ylim(-5, 5)
    elif fl_name == '/1/':
        ax1.set_xlim(-9, 1)
        ax1.set_ylim(-3, 6)
    ax1.grid()
    green_patch = mpatches.Patch(color='green', label='Target Vehicle')
    red_patch = mpatches.Patch(color='red', label='Ego Vehicle')
    ax1.legend(handles=[green_patch, red_patch])
    ax1.set_xlabel('Ego Frame x [m]')
    ax1.set_ylabel('Ego Frame y [m]')

    ax2.plot(time_[i], x[i, 0], 'o', color='green')
    if i > 0:
        ax2.plot(time_[i - 1:i + 1:1], x_pose[i - 1:i + 1:1, 0], '-', color='blue')
        ax2.plot(time_[i - 1:i + 1:1], x_roberts[i - 1:i + 1:1, 0], '-', color='purple')
        ax2.plot(time_[i - 1:i + 1:1], x_becha[i - 1:i + 1:1, 0], '-', color='orange')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('[m]')
    if fl_name == '/3/':
        ax2.set_ylim(x[0,0]-0.5,x[-1,0]+0.5)
        ax2.set_xlim(0, 10)
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
    if i > 0:
        ax3.plot(time_[i-1:i + 1:1], y_pose[i-1:i + 1:1, 0], '-', color='blue')
        ax3.plot(time_[i-1:i + 1:1], y_roberts[i-1:i + 1:1, 0], '-', color='purple')
        ax3.plot(time_[i-1:i + 1:1], y_becha[i-1:i + 1:1, 0], '-', color='orange')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('[m]')
    if fl_name == '/3/':
        ax3.set_ylim(y[0,0]-1,y[-1,0]+1)
        ax3.set_xlim(0, 10)
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

    if i == (len(x_data) -1):
        # pause
        ani.event_source.stop()

class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # tk.Tk.iconbitmap(self, default="clienticon.ico")
        tk.Tk.wm_title(self, "Plotter")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageThree):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(PageThree)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        btn_sm1 = ttk.Button(self, text="Run Simulation-1", command=partial(run, 1))
        btn_sm2 = ttk.Button(self, text="Run Simulation-2", command=partial(run, 2))
        btn_sm3 = ttk.Button(self, text="Run Simulation-3", command=partial(run, 3))
        btn_sm1.pack()
        btn_sm2.pack()
        btn_sm3.pack()



class PageThree(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # self.ani = animation.FuncAnimation(f, animate, interval=1000)
        label = tk.Label(self, text="Simulation Started", font=LARGE_FONT)
        label.pack()

        button2 = ttk.Button(self, text="Resimulate", command=partial(start_page,controller))
        button2.pack()

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


window = t.Tk()
btn_sm1 = ttk.Button(window, text="Run Simulation-1", command=partial(run, 1))
btn_sm2 = ttk.Button(window, text="Run Simulation-2", command=partial(run, 2))
btn_sm3 = ttk.Button(window, text="Run Simulation-3", command=partial(run, 3))
btn_sm1.pack()
btn_sm2.pack()
btn_sm3.pack()
image = Image.open("Figure/welcome.png")
photo = ImageTk.PhotoImage(image.resize((512, 512), Image.ANTIALIAS))
label = Label(window, image=photo)
label.image = photo
label.pack()
window.mainloop()

