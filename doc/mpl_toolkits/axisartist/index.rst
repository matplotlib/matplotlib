.. _toolkit_axisartist-index:

Matplotlib axisartist Toolkit
=============================

The *axisartist* namespace includes a derived Axes implementation. The
biggest difference is that the artists responsible to draw axis line,
ticks, ticklabel and axis labels are separated out from the mpl's Axis
class, which are much more than artists in the original mpl. This
change was strongly motivated to support curvilinear grid.

You can find a tutorial describing usage of axisartist at
:ref:`axisartist_users-guide-index`.
