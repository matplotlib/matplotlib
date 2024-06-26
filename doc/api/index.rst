API Reference
=============

Matplotlib interfaces
---------------------

Matplotlib has two interfaces. See :ref:`api_interfaces` for a more detailed
description of both and their recommended use cases.

.. grid:: 1 1 2 2
    :padding: 0
    :gutter: 2

    .. grid-item-card::
        :shadow: none
        :class-footer: api-interface-footer

        **Axes interface** (object-based, explicit)

        create a `.Figure` and one or more `~.axes.Axes` objects, then *explicitly* use
        methods on these objects to add data, configure limits, set labels etc.

        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        API:

        - `~.pyplot.subplots`: create Figure and Axes
        - :mod:`~matplotlib.axes`: add data, limits, labels etc.
        - `.Figure`: for figure-level methods

        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        Example:

        .. code-block:: python
           :class: api-interface-example

           fig, ax = plt.subplots()
           ax.plot(x, y)
           ax.set_title("Sample plot")
           plt.show()


    .. grid-item-card::
        :shadow: none
        :class-footer: api-interface-footer

        **pyplot interface** (function-based, implicit)

        consists of functions in the `.pyplot` module. Figure and Axes are manipulated
        through these functions and are only *implicitly* present in the background.

        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        API:

        - `matplotlib.pyplot`

        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        Example:

        .. code-block:: python
           :class: api-interface-example

           plt.plot(x, y)
           plt.title("Sample plot")
           plt.show()


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
   sphinxext_roles.rst
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
