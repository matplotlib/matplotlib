.. _toolkit_axisartist-index:

Matplotlib axisartist Toolkit
=============================

The *axisartist* namespace includes a derived Axes implementation (
:class:`mpl_toolkits.axisartist.Axes`). The
biggest difference is that the artists that are responsible for drawing
axis lines, ticks, ticklabels, and axis labels are separated out from the
mpl's Axis class. This change was strongly motivated to support curvilinear grid.

You can find a tutorial describing usage of axisartist at
:ref:`axisartist_users-guide-index`.
