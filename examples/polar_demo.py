#!/usr/bin/env python
#
# matplotlib now has a PolarAxes class and a polar function in the
# matplotlib interface.  This is considered alpha and the interface
# may change as we work out how polar axes should best be integrated
#
# The only function that has been tested on polar axes is "plot" (the
# matlab interface function "polar" calls ax.plot where ax is a
# PolarAxes) -- other axes plotting functions may work on PolarAxes
# but haven't been tested and may need tweaking.
#
# you can get get a PolarSubplot instance by doing, for example
#
#   subplot(211, polar=True)
#
# or a PolarAxes instance by doing
#   axes([left, bottom, width, height], polar=True)
#
# The view limits (eg xlim and ylim) apply to the lower left and upper
# right of the rectangular box that surrounds to polar axes.  Eg if
# you have
#
#  r = arange(0,1,0.01)
#  theta = 2*pi*r
#
# the lower left corner is 5/4pi, sqrt(2) and the
# upper right corner is 1/4pi, sqrt(2)
#
# you could change the radial bounding box (zoom out) by setting the
# ylim (radial coordinate is the second argument to the plot command,
# as in matlab, though this is not advised currently because it is not
# clear to me how the axes should behave in the change of view limits.
# Please advise me if you have opinions.  Likewise, the pan/zoom
# controls probably do not do what you think they do and are better
# left alone on polar axes.  Perhaps I will disable them for polar
# axes unless we come up with a meaningful, useful and functional
# implementation for them.
#
# Note that polar axes are sufficiently different that regular axes
# that I haven't stived for a consistent interface to the gridlines,
# labels, etc.  To set the properties of the gridlines and labels,
# access the attributes directly from the polar axes, as in
#
#   ax = gca()
#   set(ax.rgridlines, color='r')
#
# The following attributes are defined
#
#      thetagridlines  : a list of Line2D for the theta grids
#      rgridlines      : a list of Line2D for the radial grids
#      thetagridlabels : a list of Text for the theta grid labels
#      rgridlabels     : a list of Text for the theta grid labels                  

from matplotlib.matlab import *

r = arange(0,4,0.001)
theta = 6*pi*r
polar(theta, r)
title("It's about time!")
savefig('polar_demo')
ax = gca()

show()
