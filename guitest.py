import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import tkinter as t
import tkinter as tk
from tkinter import ttk
from tkinter import *
import matplotlib.patches as mpatches
from subprocess import call
from functools import partial
from PIL import ImageTk, Image
import time

matplotlib.use("TkAgg")
LARGE_FONT = ("Verdana", 12)
style.use("ggplot")
global a, f
f = Figure(figsize=(5, 5), dpi=100)
a = f.add_subplot(111)
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
    global fl_name
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

    #call(["python", "data_test.py", input_name])

    app = SeaofBTCapp()
    ani = animation.FuncAnimation(f, animate, interval=1000)
    app.mainloop()


def animate(i):
    global res, a, f
    if res == TRUE:
        f.clf()
        a = f.add_subplot(111)
        res = FALSE

    folder_name = 'GUI_data' + fl_name
    x, y = np.loadtxt(folder_name+'x.txt', delimiter=','), np.loadtxt(folder_name+'y.txt', delimiter=',')
    x_pose, y_pose = np.loadtxt(folder_name+'x_pose.txt', delimiter=','), np.loadtxt(folder_name+'y_pose.txt', delimiter=',')
    x_becha, y_becha = np.loadtxt(folder_name+'x_becha.txt', delimiter=','), np.loadtxt(folder_name+'y_becha.txt',
                                                                                     delimiter=',')
    x_roberts, y_roberts = np.loadtxt(folder_name+'x_roberts.txt', delimiter=','), np.loadtxt(folder_name+'y_roberts.txt',
                                                                                           delimiter=',')
    x_data, y_data = np.loadtxt(folder_name+'x_data.txt', delimiter=','), np.loadtxt(folder_name+'y_data.txt',
                                                                                  delimiter=',')

    # time.sleep(0.025)
    a.plot(x[i], y[i], 'o', color='green', markersize=10)
    a.plot(x_becha[i], y_becha[i], 'o', color='orange', markersize=8)
    a.plot(x_pose[i], y_pose[i], 'o', color='blue', markersize=7)
    a.plot(x_roberts[i], y_roberts[i], 'o', color='purple', markersize=5)
    a.plot(x_data[i], y_data[i], '--', color='red', markersize=5)
    a.grid()

    green_patch = mpatches.Patch(color='green', label='Actual coordinates')
    blue_patch = mpatches.Patch(color='blue', label='AoA-estimated coordinates')
    orange_patch = mpatches.Patch(color='orange', label='RToF-estimated coordinates')
    purple_patch = mpatches.Patch(color='purple', label='TDoA-estimated coordinates')

    a.legend(handles=[green_patch, blue_patch, orange_patch, purple_patch])
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
        label.pack(pady=100, padx=100)

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

