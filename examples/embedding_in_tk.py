from matplotlib.numerix import arange, sin, pi

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.axes import Subplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import Tkinter as Tk
import sys

def destroy(e): sys.exit()

root = Tk.Tk()
root.wm_title("Embedding in TK")
root.bind("<Destroy>", destroy)


f = Figure(figsize=(5,4), dpi=100)
a = Subplot(f, 111)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)

a.plot(t,s)
f.add_axis(a)

canvas = FigureCanvasTkAgg(f, master=root)  # a tk.DrawingArea
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP)

button = Tk.Button(master=root, text='Quit', command=sys.exit)
button.pack(side=Tk.BOTTOM)

Tk.mainloop()
