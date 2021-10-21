API Reference
=============

When using the library you will typically create
:doc:`Figure <figure_api>` and :doc:`Axes <axes_api>` objects and
call their methods to add content and modify the appearance.

- :mod:`matplotlib.figure`: axes creation, figure-level content
- :mod:`matplotlib.axes`: most plotting methods, Axes labels, access to axis
  styling, etc.

Example: We create a Figure ``fig`` and Axes ``ax``. Then we call
methods on them to plot data, add axis labels and a figure title.

.. plot::
   :include-source:
   :align: center

   import matplotlib.pyplot as plt
   import numpy as np

   x = np.arange(0, 4, 0.05)
   y = np.sin(x*np.pi)

   fig, ax = plt.subplots(figsize=(3,2), constrained_layout=True)
   ax.plot(x, y)
   ax.set_xlabel('t [s]')
   ax.set_ylabel('S [V]')
   ax.set_title('Sine wave')
   fig.set_facecolor('lightsteelblue')


Modules
-------

Alphabetical list of modules:

.. toctree::
   :maxdepth: 1

   matplotlib_configuration_api.rst
   afm_api.rst
   animation_api.rst
   artist_api.rst
   axes_api.rst
   axis_api.rst
   backend_bases_api.rst
   backend_managers_api.rst
   backend_tools_api.rst
   index_backend_api.rst
   bezier_api.rst
   blocking_input_api.rst
   category_api.rst
   cbook_api.rst
   cm_api.rst
   collections_api.rst
   colorbar_api.rst
   colors_api.rst
   container_api.rst
   contour_api.rst
   dates_api.rst
   docstring_api.rst
   dviread.rst
   figure_api.rst
   font_manager_api.rst
   fontconfig_pattern_api.rst
   gridspec_api.rst
   image_api.rst
   legend_api.rst
   legend_handler_api.rst
   lines_api.rst
   markers_api.rst
   mathtext_api.rst
   mlab_api.rst
   offsetbox_api.rst
   patches_api.rst
   path_api.rst
   patheffects_api.rst
   pyplot_summary.rst
   projections_api.rst
   quiver_api.rst
   rcsetup_api.rst
   sankey_api.rst
   scale_api.rst
   sphinxext_mathmpl_api.rst
   sphinxext_plot_directive_api.rst
   spines_api.rst
   style_api.rst
   table_api.rst
   testing_api.rst
   text_api.rst
   texmanager_api.rst
   textpath_api.rst
   ticker_api.rst
   tight_bbox_api.rst
   tight_layout_api.rst
   transformations.rst
   tri_api.rst
   type1font.rst
   units_api.rst
   widgets_api.rst
   _api_api.rst
   _enums_api.rst
   toolkits/mplot3d.rst
   toolkits/axes_grid1.rst
   toolkits/axisartist.rst
   toolkits/axes_grid.rst


.. _usage_patterns:

Usage patterns
--------------

Below we describe several common approaches to plotting with Matplotlib.

The pyplot API
^^^^^^^^^^^^^^

`matplotlib.pyplot` is a collection of functions that make
Matplotlib work like MATLAB. Each pyplot function makes some change to a
figure: e.g., creates a figure, creates a plotting area in a figure, plots
some lines in a plotting area, decorates the plot with labels, etc.

`.pyplot` is mainly intended for interactive plots and simple cases of
programmatic plot generation.

Further reading:

- The `matplotlib.pyplot` function reference
- :doc:`/tutorials/introductory/pyplot`
- :ref:`Pyplot examples <pyplots_examples>`

.. _api-index:

The object-oriented API
^^^^^^^^^^^^^^^^^^^^^^^

At its core, Matplotlib is object-oriented. We recommend directly working
with the objects, if you need more control and customization of your plots.

In many cases you will create a `.Figure` and one or more
`~matplotlib.axes.Axes` using `.pyplot.subplots` and from then on only work
on these objects. However, it's also possible to create `.Figure`\ s
explicitly (e.g. when including them in GUI applications).

Further reading:

- `matplotlib.axes.Axes` and `matplotlib.figure.Figure` for an overview of
   plotting functions.
- Most of the :ref:`examples <examples-index>` use the object-oriented approach
   (except for the pyplot section)

The pylab API (discouraged)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pylab
   :no-members:
