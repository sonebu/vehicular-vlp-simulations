import sys
import os
from tkinter import *
from subprocess import call
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename, asksaveasfilename
from functools import partial

window = Tk()
global img
name = 'Figure/' + 'welcome' + '.png'

input_name = 'v2lcRun_sm3_comparisonSoA'

def run(sm):
    # first key text
    if sm == 3:
        input_name = 'v2lcRun_sm3_comparisonSoA'
    elif sm == 2:
        input_name = 'v2lcRun_sm2_platoonFormExit'
    elif sm == 1:
        input_name = 'v2lcRun_sm1_laneChange'

    call(["python", "data_test.py", input_name])


window.rowconfigure(0, minsize=800, weight=1)
window.columnconfigure(1, minsize=800, weight=1)
img = ImageTk.PhotoImage(Image.open(name).resize((700, 900), Image.ANTIALIAS))
label = Label(window, image=img)
label.grid(row=0, column=1, sticky="nsew")

name2 = 'Figure/' + input_name + '.png'
def show_output():
    """Save the current file as a new file."""
    img2 = ImageTk.PhotoImage(Image.open(name2))
    label.configure(image=img2)
    label.image = img2


fr_buttons = Frame(window, relief=RAISED, bd=2)
btn_sm1 = Button(fr_buttons, text="Run Simulation-1",command=partial(run, 1))
btn_sm2 = Button(fr_buttons, text="Run Simulation-2",command=partial(run, 2))
btn_sm3 = Button(fr_buttons, text="Run Simulation-3",command=partial(run, 3))

btn_out = Button(fr_buttons, text="Show Output", command=show_output)

btn_sm1.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
btn_sm2.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
btn_sm3.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
btn_out.grid(row=3, column=0, sticky="ew", padx=5)

fr_buttons.grid(row=0, column=0, sticky="ns")

window.mainloop()
