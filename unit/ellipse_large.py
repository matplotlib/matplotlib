
# This example can be boiled down to a more simplistic example
# to show the problem, but including the upper and lower
# bound ellipses, it demonstrates how significant this error
# is to our plots.

from __future__ import print_function
import math
from pylab import *
from matplotlib.patches import Ellipse, Arc

# given a point x, y
x = 2692.440
y = 6720.850

# get is the radius of a circle through this point
r = math.sqrt(x * x + y * y)

# show some comparative circles
delta = 6


##################################################
def custom_ellipse(ax, x, y, major, minor, theta, numpoints=750, **kwargs):
    xs = []
    ys = []
    incr = 2.0 * math.pi / numpoints
    incrTheta = 0.0
    while incrTheta <= (2.0 * math.pi):
        a = major * math.cos(incrTheta)
        b = minor * math.sin(incrTheta)
        l = math.sqrt((a ** 2) + (b ** 2))
        phi = math.atan2(b, a)
        incrTheta += incr

        xs.append(x + (l * math.cos(theta + phi)))
        ys.append(y + (l * math.sin(theta + phi)))
    # end while

    incrTheta = 2.0 * math.pi
    a = major * math.cos(incrTheta)
    b = minor * math.sin(incrTheta)
    l = sqrt((a ** 2) + (b ** 2))
    phi = math.atan2(b, a)
    xs.append(x + (l * math.cos(theta + phi)))
    ys.append(y + (l * math.sin(theta + phi)))

    ellipseLine = ax.plot(xs, ys, **kwargs)


##################################################
# make the axes
ax1 = subplot(311, aspect='equal')
ax1.set_aspect('equal', 'datalim')

# make the lower-bound ellipse
diam = (r - delta) * 2.0
lower_ellipse = Ellipse((0.0, 0.0), diam, diam, 0.0, fill=False, edgecolor="darkgreen")
ax1.add_patch(lower_ellipse)

# make the target ellipse
diam = r * 2.0
target_ellipse = Ellipse((0.0, 0.0), diam, diam, 0.0, fill=False, edgecolor="darkred")
ax1.add_patch(target_ellipse)

# make the upper-bound ellipse
diam = (r + delta) * 2.0
upper_ellipse = Ellipse((0.0, 0.0), diam, diam, 0.0, fill=False, edgecolor="darkblue")
ax1.add_patch(upper_ellipse)

# make the target
diam = delta * 2.0
target = Ellipse((x, y), diam, diam, 0.0, fill=False, edgecolor="#DD1208")
ax1.add_patch(target)

# give it a big marker
ax1.plot([x], [y], marker='x', linestyle='None', mfc='red', mec='red', markersize=10)

##################################################
# make the axes
ax = subplot(312, aspect='equal', sharex=ax1, sharey=ax1)
ax.set_aspect('equal', 'datalim')

# make the lower-bound arc
diam = (r - delta) * 2.0
lower_arc = Arc((0.0, 0.0), diam, diam, 0.0, fill=False, edgecolor="darkgreen")
ax.add_patch(lower_arc)

# make the target arc
diam = r * 2.0
target_arc = Arc((0.0, 0.0), diam, diam, 0.0, fill=False, edgecolor="darkred")
ax.add_patch(target_arc)

# make the upper-bound arc
diam = (r + delta) * 2.0
upper_arc = Arc((0.0, 0.0), diam, diam, 0.0, fill=False, edgecolor="darkblue")
ax.add_patch(upper_arc)

# make the target
diam = delta * 2.0
target = Arc((x, y), diam, diam, 0.0, fill=False, edgecolor="#DD1208")
ax.add_patch(target)

# give it a big marker
ax.plot([x], [y], marker='x', linestyle='None', mfc='red', mec='red', markersize=10)

##################################################
# now lets do the same thing again using a custom ellipse function

# make the axes
ax = subplot(313, aspect='equal', sharex=ax1, sharey=ax1)
ax.set_aspect('equal', 'datalim')

# make the lower-bound ellipse
custom_ellipse(ax, 0.0, 0.0, r - delta, r - delta, 0.0, color="darkgreen")

# make the target ellipse
custom_ellipse(ax, 0.0, 0.0, r, r, 0.0, color="darkred")

# make the upper-bound ellipse
custom_ellipse(ax, 0.0, 0.0, r + delta, r + delta, 0.0, color="darkblue")

# make the target
custom_ellipse(ax, x, y, delta, delta, 0.0, color="#BB1208")

# give it a big marker
ax.plot([x], [y], marker='x', linestyle='None', mfc='red', mec='red', markersize=10)


# give it a big marker
ax.plot([x], [y], marker='x', linestyle='None', mfc='red', mec='red', markersize=10)

##################################################
# lets zoom in to see the area of interest

ax1.set_xlim(2650, 2735)
ax1.set_ylim(6705, 6735)

savefig("ellipse")
show()
