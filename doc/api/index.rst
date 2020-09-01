API Overview
============

.. toctree::
   :hidden:

   api_changes

.. contents:: :local:

See also the :doc:`api_changes`.

Usage patterns
--------------

Below we describe several common approaches to plotting with Matplotlib.

The pyplot API
^^^^^^^^^^^^^^

`matplotlib.pyplot` is a collection of command style functions that make
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

The pylab API (disapproved)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pylab
   :no-members:

Modules
-------

Matplotlib consists of the following submodules:

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
   sphinxext_plot_directive_api.rst
   spines_api.rst
   style_api.rst
   table_api.rst
   testing_api.rst
   text_api.rst
   texmanager_api.rst
   textpath_api.rst
   ticker_api.rst
   tight_layout_api.rst
   transformations.rst
   tri_api.rst
   type1font.rst
   units_api.rst
   widgets_api.rst

Toolkits
--------

:ref:`toolkits-index` are collections of application-specific functions that extend
Matplotlib. The following toolkits are included:

.. toctree::
   :hidden:

   toolkits/index.rst

.. toctree::
   :maxdepth: 1

   toolkits/mplot3d.rst
   toolkits/axes_grid1.rst
   toolkits/axisartist.rst
   toolkits/axes_grid.rst
