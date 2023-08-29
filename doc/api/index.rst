API Reference
=============

Matplotlib interfaces
---------------------

Matplotlib provides two different interfaces:

- **Axes interface**: create a `.Figure` and one or more `~.axes.Axes` objects
  (typically using `.pyplot.subplots`), then *explicitly* use methods on these objects
  to add data, configure limits, set labels etc.
- **pyplot interface**: consists of functions in the `.pyplot` module. Figure and Axes
  are manipulated through these functions and are only *implicitly* present in the
  background.

See :ref:`api_interfaces` for a more detailed description of both and their recommended
use cases.

.. grid:: 1 1 2 2
    :padding: 0 0 1 1

    .. grid-item-card::

        **Axes interface** (object-based, explicit)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        - `~.pyplot.subplots` or `~.pyplot.subplot_mosaic`: create Figure and Axes
        - `Axes <matplotlib.axes>`: add data, limits, labels etc.
        - `.Figure`: for figure-level methods

    .. grid-item-card::

        **pyplot interface** (function-based, implicit)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        - `matplotlib.pyplot`

.. tab-set::

  .. tab-item:: Axes interface

    .. plot::
      :include-source:
      :align: center

      x = np.arange(0, 4, 0.05)
      y = np.sin(x*np.pi)
      # Create a Figure and an Axes:
      fig, ax = plt.subplots(figsize=(3,2), layout='constrained')
      # Use the Axes to plot and label:
      ax.plot(x, y)
      ax.set_xlabel('t [s]')
      ax.set_ylabel('S [V]')
      ax.set_title('Sine wave')
      # Change a property of the Figure:
      fig.set_facecolor('lightsteelblue')


  .. tab-item:: pyplot interface

    .. plot::
      :include-source:
      :align: center

      x = np.arange(0, 4, 0.05)
      y = np.sin(x*np.pi)
      # Create a Figure of a given size:
      plt.figure(figsize=(3, 2), layout='constrained')
      # plot on a default Axes on the Figure:
      plt.plot(x, y)
      plt.xlabel('t [s]')
      plt.ylabel('S [V]')
      plt.title('Sine wave')
      # set a Figure property:
      plt.gcf().set_facecolor('lightsteelblue')


.. _api-index:

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
   ft2font.rst
   gridspec_api.rst
   hatch_api.rst
   image_api.rst
   layout_engine_api.rst
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
   sphinxext_figmpl_directive_api.rst
   spines_api.rst
   style_api.rst
   table_api.rst
   testing_api.rst
   text_api.rst
   texmanager_api.rst
   ticker_api.rst
   tight_bbox_api.rst
   tight_layout_api.rst
   transformations.rst
   tri_api.rst
   type1font.rst
   typing_api.rst
   units_api.rst
   widgets_api.rst
   _api_api.rst
   _enums_api.rst
   toolkits/mplot3d.rst
   toolkits/axes_grid1.rst
   toolkits/axisartist.rst
   pylab.rst
