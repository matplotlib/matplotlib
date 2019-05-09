:orphan:

Compass notation for legend and other anchored artists
------------------------------------------------------

The ``loc`` parameter for legends and other anchored artists now accepts
"compass" strings. E.g. to locate such element in the upper right corner,
in addition to ``'upper right'`` and ``1``, you can now use ``'NE'`` as
well as ``'northeast'``. This satisfies the wish for more intuitive and
unambiguous location of legends. The following (case-sensitive) location
specifications are now allowed.

    ============  ==============  ===============  =============
    Compass Code  Compass String  Location String  Location Code
    ============  ==============  ===============  =============
    ..                            'best'           0
    'NE'          'northeast'     'upper right'    1
    'NW'          'northwest'     'upper left'     2
    'SW'          'southwest'     'lower left'     3
    'SE'          'southeast'     'lower right'    4
    ..                            'right'          5
    'W'           'west'          'center left'    6
    'E'           'east'          'center right'   7
    'S'           'south'         'lower center'   8
    'N'           'north'         'upper center'   9
    'C'           'center'        'center'         10
    ============  ==============  ===============  =============

Those apply to 

  * the axes legends; `matplotlib.pyplot.legend` and
    `matplotlib.axes.Axes.legend`,

and, with the exception of ``'best'`` and ``0``, to

  * the figure legends; `matplotlib.pyplot.figlegend` and
    `matplotlib.figure.Figure.legend`, as well as the general
    `matplotlib.legend.Legend` class,
  * the `matplotlib.offsetbox`'s `matplotlib.offsetbox.AnchoredOffsetbox` and
    `matplotlib.offsetbox.AnchoredText`,
  * the `mpl_toolkits.axes_grid1.anchored_artists`'s
    `~.AnchoredDrawingArea`, `~.AnchoredAuxTransformBox`,
    `~.AnchoredEllipse`, `~.AnchoredSizeBar`, `~.AnchoredDirectionArrows`
  * the `mpl_toolkits.axes_grid1.inset_locator`'s 
    `~.axes_grid1.inset_locator.inset_axes`,
    `~.axes_grid1.inset_locator.zoomed_inset_axes` and the 
    `~.axes_grid1.inset_locator.AnchoredSizeLocator` and
    `~.axes_grid1.inset_locator.AnchoredZoomLocator`

Note that those new compass strings *do not* apply to ``table``.


Getter/setter for legend and other anchored artists location
------------------------------------------------------------
 
The above mentioned classes (in particular `~.legend.Legend`,
`~.offsetbox.AnchoredOffsetbox`, `~.offsetbox.AnchoredText` etc.)
now have a getter/setter for the location.
This allows to e.g. change the location *after* creating a legend::

    legend = ax.legend(loc="west")
    legend.set_loc("southeast")
