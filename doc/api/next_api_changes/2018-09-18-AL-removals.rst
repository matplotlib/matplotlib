API removals
````````````

The following deprecated APIs were removed:

- ``Verbose``,
- ``artist.Artist.hitlist``, ``artist.Artist.is_figure_set``,
- ``axis.Axis.unit_data``,
- ``backend_bases.FigureCanvasBase.onRemove``,
  ``backend_bases.FigureManagerBase.show_popup``,
- ``backend_wx.SubplotToolWx``, ``backend_wx.Toolbar``,
- ``cbook.align_iterators``,
- ``contour.ContourLabeler.get_real_label_width``,
- ``legend.Legend.draggable``,
- ``texmanager.TexManager.postscriptd``, ``texmanager.TexManager.pscnt``,
  ``texmanager.TexManager.make_ps``, ``texmanager.TexManager.get_ps_bbox``,
- the ``normed`` kwarg to ``Axes.hist``,
- the ``fig`` kwarg to ``GridSpec.get_subplot_params`` and
  ``GridSpecFromSubplotSpec.get_subplot_params``,
- svgfont support (in :rc:`svg.fonttype`),
- passing 'box-forced' to `axes.Axes.set_adjustable`,
- support for the strings 'on'/'true'/'off'/'false' to mean True/False (the
  following functions are affected: `Axes.grid`, `Axes3D.grid`,
  `Axis.set_tick_params`, `pyplot.box`),
