"""
Plot with radians from the basic_units mockup example package
This example shows how the unit class can determine the tick locating,
formatting and axis labeling.
"""
import numpy as np
from basic_units import radians, degrees, cos
from matplotlib.pyplot import figure, show

x = np.arange(0, 15, 0.01) * radians

fig = figure()
fig.subplots_adjust(hspace=0.3)

ax = fig.add_subplot(211)
ax.plot(x, cos(x), xunits=radians)

ax = fig.add_subplot(212)
ax.plot(x, cos(x), xunits=degrees)

show()
