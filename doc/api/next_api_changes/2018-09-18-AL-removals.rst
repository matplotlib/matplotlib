API removals
````````````

The following deprecated APIs were removed:

Classes and methods
-------------------
- ``Verbose`` (replaced by python logging library)
- ``artist.Artist.hitlist`` (no replacement)
- ``artist.Artist.is_figure_set`` (use ``artist.figure is not None`` instead)
- ``axis.Axis.unit_data`` (use ``axis.Axis.units`` instead)
- ``backend_bases.FigureCanvasBase.onRemove`` (no replacement)
  ``backend_bases.FigureManagerBase.show_popup`` (this never did anything)
- ``backend_wx.SubplotToolWx`` (no replacement)
- ``backend_wx.Toolbar`` (use ``backend_wx.NavigationToolbar2Wx`` instead)
- ``cbook.align_iterators`` (no replacment)
- ``contour.ContourLabeler.get_real_label_width`` (no replacement)
- ``legend.Legend.draggable`` (use `legend.Legend.set_draggable()` instead)
- ``texmanager.TexManager.postscriptd``, ``texmanager.TexManager.pscnt``,
  ``texmanager.TexManager.make_ps``, ``texmanager.TexManager.get_ps_bbox``
  (no replacements)

Arguments
---------
- The ``fig`` kwarg to ``GridSpec.get_subplot_params`` and
  ``GridSpecFromSubplotSpec.get_subplot_params`` (use the argument
  ``figure`` instead)
- Passing 'box-forced' to `axes.Axes.set_adjustable` (use 'box' instead)
- Support for the strings 'on'/'true'/'off'/'false' to mean
  ``True``/``False`` (directly use ``True``/``False`` instead).
  The following functions are affected: `Axes.grid`, `Axes3D.grid`
  `Axis.set_tick_params`, `pyplot.box`.
- Using `pyplot.axes` with an `axes.Axes` type argument
  (use `pyplot.sca` instead)

Other
-----
- svgfont support (in :rc:`svg.fonttype`) has been removed,
- Logging is now done with the standard python ``logging`` library.
  ``matplotlib.verbose`` and the command line switches ``--verbose-LEVEL`` are
  removed.

  To control the logging output use::

    import logging
    logger = logging.getLogger('matplotlib')
    logger.set_level(logging.INFO)
