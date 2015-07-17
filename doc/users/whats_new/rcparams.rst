Added ``errorbar.capsize`` key to rcParams
``````````````````````````````````````````
Controls the length of end caps on error bars. If set to zero, errorbars
are turned off by default.

Added ``xtick.minor.visible`` and ``ytick.minor.visible`` key to rcParams
`````````````````````````````````````````````````````````````````````````
Two new keys to control the minor ticks on x/y axis respectively, default set to ``False`` (no minor ticks on the axis).

When ``True``, the minor ticks are shown and located via a ``mticker.AutoMinorLocator()``.

Added "legend.framealpha" key to rcParams
`````````````````````````````````````````
Added a key and the corresponding logic to control the default transparency of
legend frames. This feature was written into the docstring of axes.legend(),
but not yet implemented.

Added "figure.titlesize" and "figure.titleweight" keys to rcParams
``````````````````````````````````````````````````````````````````

Two new keys were added to rcParams to control the default font size
and weight used by the figure title (as emitted by
``pyplot.suptitle()``).

Added ``legend.facecolor`` and ``legend.edgecolor`` keys to rcParams
```````````````````````````````````````````````````````````````````

The new keys control colors (background and edge) of legend patches.
The value ``'inherit'`` for these rcParams falls uses the value of
``axes.facecolor`` and ``axes.edgecolor``.


``image.composite_image`` added to rcParams
```````````````````````````````````````````
Controls whether vector graphics backends (i.e. PDF, PS, and SVG) combine
multiple images on a set of axes into a single composite image.  Saving each
image individually can be useful if you generate vector graphics files in
matplotlib and then edit the files further in Inkscape or other programs.

Added ``markers.fillstyle`` key to rcParams
```````````````````````````````````````````
Controls the default fillstyle of markers. Possible values are ``'full'`` (the
default), ``'left'``, ``'right'``, ``'bottom'``, ``'top'`` and ``'none'``.

Added "toolmanager" to "toolbar" possible values
````````````````````````````````````````````````

The new value enables the use of ``ToolManager``


Added ``axes.labelpad``
```````````````````````

This new value controls the space between the axis and the label
