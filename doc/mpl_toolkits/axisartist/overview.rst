==============================
Overview of axisartist toolkit
==============================

.. warning:: 
   *axisartist* uses a custom Axes class 
   (derived from the mpl's original Axes class).
   As a side effect, some commands (mostly tick-related) do not work.


The *axisartist* contains custom Axes class that is meant to support for
curvilinear grids (e.g., the world coordinate system in astronomy).
Unlike mpl's original Axes class which uses Axes.xaxis and Axes.yaxis
to draw ticks, ticklines and etc., Axes in axisartist uses special
artist (AxisArtist) which can handle tick, ticklines and etc. for
curved coordinate systems.

.. plot:: mpl_toolkits/axisartist/examples/demo_floating_axis.py

Since it uses special artists, some mpl commands that work on
Axes.xaxis and Axes.yaxis may not work.

axisartist
----------

*axisartist* module provides a custom (and very experimental) Axes
class, where each axis (left, right, top and bottom) have a separate
associated artist which is responsible to draw axis-line, ticks,
ticklabels, label.  Also, you can create your own axis, which can pass
through a fixed position in the axes coordinate, or a fixed position
in the data coordinate (i.e., the axis floats around when viewlimit
changes).

The axes class, by default, has its xaxis and yaxis invisible, and
has 4 additional artists which are responsible for drawing the 4 axis sides in
"left","right","bottom" and "top".  They are accessed as
ax.axis["left"], ax.axis["right"], and so on, i.e., ax.axis is a
dictionary that contains artists (note that ax.axis is still a
callable methods and it behaves as an original Axes.axis method in
mpl).

To create an axes, ::

  import mpl_toolkits.axisartist as AA
  fig = plt.figure(1)
  ax = AA.Axes(fig, [0.1, 0.1, 0.8, 0.8])
  fig.add_axes(ax)

or to create a subplot ::

  ax = AA.Subplot(fig, 111)
  fig.add_subplot(ax)

For example, you can hide the right, and top axis by ::

  ax.axis["right"].set_visible(False)
  ax.axis["top"].set_visible(False)


.. plot:: mpl_toolkits/axisartist/figures/simple_axisline3.py


It is also possible to add an extra axis. For example, you may have an
horizontal axis at y=0 (in data coordinate). ::

    ax.axis["y=0"] = ax.new_floating_axis(nth_coord=0, value=0)

.. plot:: mpl_toolkits/axisartist/figures/simple_axisartist1.py
   :include-source:


Or a fixed axis with some offset ::

    # make new (right-side) yaxis, but wth some offset
    ax.axis["right2"] = ax.new_fixed_axis(loc="right",
				          offset=(20, 0))



axisartist with ParasiteAxes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most commands in the axes_grid1 toolkit can take a axes_class keyword
argument, and the commands creates an axes of the given class. For example,
to create a host subplot with axisartist.Axes, ::

  import mpl_tookits.axisartist as AA
  from mpl_toolkits.axes_grid1 import host_subplot

  host = host_subplot(111, axes_class=AA.Axes)


Here is an example that uses  parasiteAxes.


.. plot:: mpl_toolkits/axisartist/examples/demo_parasite_axes2.py



Curvilinear Grid
----------------

The motivation behind the AxisArtist module is to support curvilinear grid
and ticks.

.. plot:: mpl_toolkits/axisartist/examples/demo_curvelinear_grid.py

See :ref:`axisartist-manual` for more details.


Floating Axes
-------------

This also support a Floating Axes whose outer axis are defined as
floating axis.

.. plot:: mpl_toolkits/axisartist/examples/demo_floating_axes.py
