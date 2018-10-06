API removals
````````````

The following deprecated APIs were removed:

Classes and methods
-------------------
- ``Verbose``
- ``artist.Artist.hitlist``
- ``artist.Artist.is_figure_set`` (use ``artist.figure is not None`` instead)
- ``axis.Axis.unit_data`` (use ``axis.Axis.units`` instead)
- ``backend_bases.FigureCanvasBase.onRemove``
  ``backend_bases.FigureManagerBase.show_popup`` (this never did anything)
- ``backend_wx.SubplotToolWx``, ``backend_wx.Toolbar``
- ``cbook.align_iterators``
- ``contour.ContourLabeler.get_real_label_width``
- ``legend.Legend.draggable`` (use `legend.Legend.set_draggable()` instead)
- ``texmanager.TexManager.postscriptd``, ``texmanager.TexManager.pscnt``,
  ``texmanager.TexManager.make_ps``, ``texmanager.TexManager.get_ps_bbox``

Arguments
---------
- The ``fig`` kwarg to ``GridSpec.get_subplot_params`` and
  ``GridSpecFromSubplotSpec.get_subplot_params``
- Passing 'box-forced' to `axes.Axes.set_adjustable`
- Support for the strings 'on'/'true'/'off'/'false' to mean ``True``/``False``
  the following functions are affected: `Axes.grid`, `Axes3D.grid`
  `Axis.set_tick_params`, `pyplot.box`)
- Using `pyplot.axes` with an `axes.Axes` type argument
  (use `pyplot.sca` instead)

Other
-----
- svgfont support (in :rc:`svg.fonttype`),
