Add fill_bar argument to ``AnchoredSizeBar``
--------------------------------------------

The mpl_toolkits class
:class:`~mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar` now has an
additional ``fill_bar`` argument, which makes the size bar a solid rectangle
instead of just drawing the border of the rectangle. The default is ``None``,
and whether or not the bar will be filled by default depends on the value of
``size_vertical``. If ``size_vertical`` is nonzero, ``fill_bar`` will be set to
``True``. If ``size_vertical`` is zero then ``fill_bar`` will be set to
``False``. If you wish to override this default behavior, set ``fill_bar`` to
``True`` or ``False`` to unconditionally always or never use a filled patch
rectangle for the size bar.
