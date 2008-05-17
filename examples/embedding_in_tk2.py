#!/usr/bin/env python
import Tkinter as Tk
import numpy as np
import matplotlib.backends.backend_tkagg as backend
import matplotlib.figure as mfigure


root = Tk.Tk()
root.wm_title("Embedding in TK")

fig = mfigure.Figure(figsize=(5,4), dpi=100)
ax = fig.add_subplot(111)
t = np.arange(0.0,3.0,0.01)
s = np.sin(2*np.pi*t)

ax.plot(t,s)
ax.grid(True)
ax.set_title('Tk embedding')
ax.set_xlabel('time (s)')
ax.set_ylabel('volts (V)')


# a tk.DrawingArea
canvas = backend.FigureCanvasTkAgg(fig, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

#toolbar = backend.NavigationToolbar2TkAgg( canvas, root )
#toolbar.update()
#toolbar.pack(side=Tk.LEFT)

def destroy():
    raise SystemExit

button = Tk.Button(master=root, text='Quit', command=destroy)
button.pack(side=Tk.BOTTOM)

Tk.mainloop()
