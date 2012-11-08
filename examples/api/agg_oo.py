#!/usr/bin/env python
# -*- noplot -*-
"""
A pure OO (look Ma, no pylab!) example using the agg backend
"""
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

fig, ax = plt.subplots()
ax.plot([1,2,3])
ax.set_title('hi mom')
ax.grid(True)
ax.set_xlabel('time')
ax.set_ylabel('volts')
fig.canvas.print_figure('test')
