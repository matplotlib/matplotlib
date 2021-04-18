.. module:: mpl_toolkits.axisartist

Matplotlib axisartist Toolkit
=============================

The *axisartist* namespace includes a derived Axes implementation (
:class:`mpl_toolkits.axisartist.Axes`). The
biggest difference is that the artists that are responsible for drawing
axis lines, ticks, ticklabels, and axis labels are separated out from the
mpl's Axis class. This change was strongly motivated to support curvilinear grid.

You can find a tutorial describing usage of axisartist at the 
:ref:`axisartist_users-guide-index` user guide.

.. figure:: ../../gallery/axisartist/images/sphx_glr_demo_curvelinear_grid_001.png
   :target: ../../gallery/axisartist/demo_curvelinear_grid.html
   :align: center
   :scale: 50

.. currentmodule:: mpl_toolkits

**The submodules of the axisartist API are:**

.. autosummary::
   :toctree: ../_as_gen
   :template: automodule.rst

   axisartist.angle_helper
   axisartist.axes_divider
   axisartist.axes_grid
   axisartist.axes_rgb
   axisartist.axis_artist
   axisartist.axisline_style
   axisartist.axislines
   axisartist.clip_path
   axisartist.floating_axes
   axisartist.grid_finder
   axisartist.grid_helper_curvelinear
   axisartist.parasite_axes

