"""
==========================
Sample plots in Matplotlib
==========================

Here you'll find a host of example plots with the code that
generated them.

.. _matplotlibscreenshots:

Line Plot
=========

Here's how to create a line plot with text labels using
:func:`~matplotlib.pyplot.plot`.

.. figure:: ../../gallery/lines_bars_and_markers/images/sphx_glr_simple_plot_001.png
   :target: ../../gallery/lines_bars_and_markers/simple_plot.html
   :align: center
   :scale: 50

   Simple Plot

.. _screenshots_subplot_demo:

Multiple subplots in one figure
===============================

Multiple axes (i.e. subplots) are created with the
:func:`~matplotlib.pyplot.subplot` function:

.. figure:: ../../gallery/subplots_axes_and_figures/images/sphx_glr_subplot_001.png
   :target: ../../gallery/subplots_axes_and_figures/subplot.html
   :align: center
   :scale: 50

   Subplot

.. _screenshots_images_demo:

Images
======

Matplotlib can display images (assuming equally spaced
horizontal dimensions) using the :func:`~matplotlib.pyplot.imshow` function.

.. figure:: ../../gallery/images_contours_and_fields/images/sphx_glr_image_demo_003.png
   :target: ../../gallery/images_contours_and_fields/image_demo.html
   :align: center
   :scale: 50

   Example of using :func:`~matplotlib.pyplot.imshow` to display a CT scan

.. _screenshots_pcolormesh_demo:


Contouring and pseudocolor
==========================

The :func:`~matplotlib.pyplot.pcolormesh` function can make a colored
representation of a two-dimensional array, even if the horizontal dimensions
are unevenly spaced.  The
:func:`~matplotlib.pyplot.contour` function is another way to represent
the same data:

.. figure:: ../../gallery/images_contours_and_fields/images/sphx_glr_pcolormesh_levels_001.png
   :target: ../../gallery/images_contours_and_fields/pcolormesh_levels.html
   :align: center
   :scale: 50

   Example comparing :func:`~matplotlib.pyplot.pcolormesh` and :func:`~matplotlib.pyplot.contour` for plotting two-dimensional data

.. _screenshots_histogram_demo:

Histograms
==========

The :func:`~matplotlib.pyplot.hist` function automatically generates
histograms and returns the bin counts or probabilities:

.. figure:: ../../gallery/statistics/images/sphx_glr_histogram_features_001.png
   :target: ../../gallery/statistics/histogram_features.html
   :align: center
   :scale: 50

   Histogram Features


.. _screenshots_path_demo:

Paths
=====

You can add arbitrary paths in Matplotlib using the
:mod:`matplotlib.path` module:

.. figure:: ../../gallery/shapes_and_collections/images/sphx_glr_path_patch_001.png
   :target: ../../gallery/shapes_and_collections/path_patch.html
   :align: center
   :scale: 50

   Path Patch

.. _screenshots_mplot3d_surface:

Three-dimensional plotting
==========================

The mplot3d toolkit (see :ref:`toolkit_mplot3d-tutorial` and
:ref:`mplot3d-examples-index`) has support for simple 3d graphs
including surface, wireframe, scatter, and bar charts.

.. figure:: ../../gallery/mplot3d/images/sphx_glr_surface3d_001.png
   :target: ../../gallery/mplot3d/surface3d.html
   :align: center
   :scale: 50

   Surface3d

Thanks to John Porter, Jonathon Taylor, Reinier Heeres, and Ben Root for
the `.mplot3d` toolkit. This toolkit is included with all standard Matplotlib
installs.

.. _screenshots_ellipse_demo:


Streamplot
==========

The :meth:`~matplotlib.pyplot.streamplot` function plots the streamlines of
a vector field. In addition to simply plotting the streamlines, it allows you
to map the colors and/or line widths of streamlines to a separate parameter,
such as the speed or local intensity of the vector field.

.. figure:: ../../gallery/images_contours_and_fields/images/sphx_glr_plot_streamplot_001.png
   :target: ../../gallery/images_contours_and_fields/plot_streamplot.html
   :align: center
   :scale: 50

   Streamplot with various plotting options.

This feature complements the :meth:`~matplotlib.pyplot.quiver` function for
plotting vector fields. Thanks to Tom Flannaghan and Tony Yu for adding the
streamplot function.


Ellipses
========

In support of the `Phoenix <http://www.jpl.nasa.gov/news/phoenix/main.php>`_
mission to Mars (which used Matplotlib to display ground tracking of
spacecraft), Michael Droettboom built on work by Charlie Moad to provide
an extremely accurate 8-spline approximation to elliptical arcs (see
:class:`~matplotlib.patches.Arc`), which are insensitive to zoom level.

.. figure:: ../../gallery/shapes_and_collections/images/sphx_glr_ellipse_demo_001.png
   :target: ../../gallery/shapes_and_collections/ellipse_demo.html
   :align: center
   :scale: 50

   Ellipse Demo

.. _screenshots_barchart_demo:

Bar charts
==========

Use the :func:`~matplotlib.pyplot.bar` function to make bar charts, which
includes customizations such as error bars:

.. figure:: ../../gallery/statistics/images/sphx_glr_barchart_demo_001.png
   :target: ../../gallery/statistics/barchart_demo.html
   :align: center
   :scale: 50

   Barchart Demo

You can also create stacked bars
(`bar_stacked.py <../../gallery/lines_bars_and_markers/bar_stacked.html>`_),
or horizontal bar charts
(`barh.py <../../gallery/lines_bars_and_markers/barh.html>`_).

.. _screenshots_pie_demo:


Pie charts
==========

The :func:`~matplotlib.pyplot.pie` function allows you to create pie
charts.  Optional features include auto-labeling the percentage of area,
exploding one or more wedges from the center of the pie, and a shadow effect.
Take a close look at the attached code, which generates this figure in just
a few lines of code.

.. figure:: ../../gallery/pie_and_polar_charts/images/sphx_glr_pie_features_001.png
   :target: ../../gallery/pie_and_polar_charts/pie_features.html
   :align: center
   :scale: 50

   Pie Features

.. _screenshots_table_demo:

Tables
======

The :func:`~matplotlib.pyplot.table` function adds a text table
to an axes.

.. figure:: ../../gallery/misc/images/sphx_glr_table_demo_001.png
   :target: ../../gallery/misc/table_demo.html
   :align: center
   :scale: 50

   Table Demo


.. _screenshots_scatter_demo:


Scatter plots
=============

The :func:`~matplotlib.pyplot.scatter` function makes a scatter plot
with (optional) size and color arguments. This example plots changes
in Google's stock price, with marker sizes reflecting the
trading volume and colors varying with time. Here, the
alpha attribute is used to make semitransparent circle markers.

.. figure:: ../../gallery/lines_bars_and_markers/images/sphx_glr_scatter_demo2_001.png
   :target: ../../gallery/lines_bars_and_markers/scatter_demo2.html
   :align: center
   :scale: 50

   Scatter Demo2


.. _screenshots_slider_demo:

GUI widgets
===========

Matplotlib has basic GUI widgets that are independent of the graphical
user interface you are using, allowing you to write cross GUI figures
and widgets.  See :mod:`matplotlib.widgets` and the
`widget examples <../../gallery/index.html>`_.

.. figure:: ../../gallery/widgets/images/sphx_glr_slider_demo_001.png
   :target: ../../gallery/widgets/slider_demo.html
   :align: center
   :scale: 50

   Slider and radio-button GUI.


.. _screenshots_fill_demo:

Filled curves
=============

The :func:`~matplotlib.pyplot.fill` function lets you
plot filled curves and polygons:

.. figure:: ../../gallery/lines_bars_and_markers/images/sphx_glr_fill_001.png
   :target: ../../gallery/lines_bars_and_markers/fill.html
   :align: center
   :scale: 50

   Fill

Thanks to Andrew Straw for adding this function.

.. _screenshots_date_demo:

Date handling
=============

You can plot timeseries data with major and minor ticks and custom
tick formatters for both.

.. figure:: ../../gallery/text_labels_and_annotations/images/sphx_glr_date_001.png
   :target: ../../gallery/text_labels_and_annotations/date.html
   :align: center
   :scale: 50

   Date

See :mod:`matplotlib.ticker` and :mod:`matplotlib.dates` for details and usage.


.. _screenshots_log_demo:

Log plots
=========

The :func:`~matplotlib.pyplot.semilogx`,
:func:`~matplotlib.pyplot.semilogy` and
:func:`~matplotlib.pyplot.loglog` functions simplify the creation of
logarithmic plots.

.. figure:: ../../gallery/scales/images/sphx_glr_log_demo_001.png
   :target: ../../gallery/scales/log_demo.html
   :align: center
   :scale: 50

   Log Demo

Thanks to Andrew Straw, Darren Dale and Gregory Lielens for contributions
log-scaling infrastructure.

.. _screenshots_polar_demo:

Polar plots
===========

The :func:`~matplotlib.pyplot.polar` function generates polar plots.

.. figure:: ../../gallery/pie_and_polar_charts/images/sphx_glr_polar_demo_001.png
   :target: ../../gallery/pie_and_polar_charts/polar_demo.html
   :align: center
   :scale: 50

   Polar Demo

.. _screenshots_legend_demo:


Legends
=======

The :func:`~matplotlib.pyplot.legend` function automatically
generates figure legends, with MATLAB-compatible legend-placement
functions.

.. figure:: ../../gallery/text_labels_and_annotations/images/sphx_glr_legend_001.png
   :target: ../../gallery/text_labels_and_annotations/legend.html
   :align: center
   :scale: 50

   Legend

Thanks to Charles Twardy for input on the legend function.

.. _screenshots_mathtext_examples_demo:

TeX-notation for text objects
=============================

Below is a sampling of the many TeX expressions now supported by Matplotlib's
internal mathtext engine.  The mathtext module provides TeX style mathematical
expressions using `FreeType <https://www.freetype.org/>`_
and the DejaVu, BaKoMa computer modern, or `STIX <http://www.stixfonts.org>`_
fonts.  See the :mod:`matplotlib.mathtext` module for additional details.

.. figure:: ../../gallery/text_labels_and_annotations/images/sphx_glr_mathtext_examples_001.png
   :target: ../../gallery/text_labels_and_annotations/mathtext_examples.html
   :align: center
   :scale: 50

   Mathtext Examples

Matplotlib's mathtext infrastructure is an independent implementation and
does not require TeX or any external packages installed on your computer. See
the tutorial at :doc:`/tutorials/text/mathtext`.


.. _screenshots_tex_demo:

Native TeX rendering
====================

Although Matplotlib's internal math rendering engine is quite
powerful, sometimes you need TeX. Matplotlib supports external TeX
rendering of strings with the *usetex* option.

.. figure:: ../../gallery/text_labels_and_annotations/images/sphx_glr_tex_demo_001.png
   :target: ../../gallery/text_labels_and_annotations/tex_demo.html
   :align: center
   :scale: 50

   Tex Demo

.. _screenshots_eeg_demo:

EEG GUI
=======

You can embed Matplotlib into pygtk, wx, Tk, or Qt applications.
Here is a screenshot of an EEG viewer called `pbrain
<https://github.com/nipy/pbrain>`__.

.. image:: ../../_static/eeg_small.png

The lower axes uses :func:`~matplotlib.pyplot.specgram`
to plot the spectrogram of one of the EEG channels.

For examples of how to embed Matplotlib in different toolkits, see:

   * :doc:`/gallery/user_interfaces/embedding_in_gtk3_sgskip`
   * :doc:`/gallery/user_interfaces/embedding_in_wx2_sgskip`
   * :doc:`/gallery/user_interfaces/mpl_with_glade3_sgskip`
   * :doc:`/gallery/user_interfaces/embedding_in_qt_sgskip`
   * :doc:`/gallery/user_interfaces/embedding_in_tk_sgskip`

XKCD-style sketch plots
=======================

Just for fun, Matplotlib supports plotting in the style of `xkcd
<https://www.xkcd.com/>`_.

.. figure:: ../../gallery/showcase/images/sphx_glr_xkcd_001.png
   :target: ../../gallery/showcase/xkcd.html
   :align: center
   :scale: 50

   xkcd

Subplot example
===============

Many plot types can be combined in one figure to create
powerful and flexible representations of data.
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
data = np.random.randn(2, 100)

fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs[0, 0].hist(data[0])
axs[1, 0].scatter(data[0], data[1])
axs[0, 1].plot(data[0], data[1])
axs[1, 1].hist2d(data[0], data[1])

plt.show()
