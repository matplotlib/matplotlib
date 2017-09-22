Removal of deprecated features
``````````````````````````````

The ``matplotlib.finance``, ``mpl_toolkits.exceltools`` and
``mpl_toolkits.gtktools`` modules have been removed.  ``matplotlib.finance``
remains available at https://github.com/matplotlib/mpl_finance.

The ``mpl_toolkits.mplot3d.art3d.iscolor`` function has been removed.

The ``Axes.get_axis_bgcolor``, ``Axes.set_axis_bgcolor``,
``Bbox.update_from_data``, ``Bbox.update_datalim_numerix``,
``MaxNLocator.bin_boundaries`` methods have been removed.

The ``bgcolor`` keyword argument to ``Axes`` has been removed.

The ``spectral`` colormap has been removed.  The ``Vega*`` colormaps, which
were aliases for the ``tab*`` colormaps, have been removed.
