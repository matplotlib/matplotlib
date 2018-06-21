"""
=====================================================
Plotting datasets in different units on the same axes
=====================================================

This example showcases how to plot two different related datasets with
different units on the same axes. Here, we plot the density and temperature as
a function of the distance on simulated data. The density y-axis is displayed
on the left hand-side of the plot, while the temperature y-axis is on the
right hand-side of the plot.

"""
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.pyplot as plt

host = host_subplot(111)

par = host.twinx()

host.set_xlabel("Distance")
host.set_ylabel("Density")
par.set_ylabel("Temperature")

p1, = host.plot([0, 1, 2], [0, 1, 2], label="Density")
p2, = par.plot([0, 1, 2], [0, 3, 2], label="Temperature")

leg = plt.legend()

host.yaxis.get_label().set_color(p1.get_color())
leg.texts[0].set_color(p1.get_color())

par.yaxis.get_label().set_color(p2.get_color())
leg.texts[1].set_color(p2.get_color())

plt.show()
