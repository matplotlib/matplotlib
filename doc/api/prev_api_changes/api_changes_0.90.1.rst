
Changes for 0.90.1
==================

.. code-block:: text

    The file dviread.py has a (very limited and fragile) dvi reader
    for usetex support. The API might change in the future so don't
    depend on it yet.

    Removed deprecated support for a float value as a gray-scale;
    now it must be a string, like '0.5'.  Added alpha kwarg to
    ColorConverter.to_rgba_list.

    New method set_bounds(vmin, vmax) for formatters, locators sets
    the viewInterval and dataInterval from floats.

    Removed deprecated colorbar_classic.

    Line2D.get_xdata and get_ydata valid_only=False kwarg is replaced
    by orig=True.  When True, it returns the original data, otherwise
    the processed data (masked, converted)

    Some modifications to the units interface.
    units.ConversionInterface.tickers renamed to
    units.ConversionInterface.axisinfo and it now returns a
    units.AxisInfo object rather than a tuple.  This will make it
    easier to add axis info functionality (e.g., I added a default label
    on this iteration) w/o having to change the tuple length and hence
    the API of the client code every time new functionality is added.
    Also, units.ConversionInterface.convert_to_value is now simply
    named units.ConversionInterface.convert.

    Axes.errorbar uses Axes.vlines and Axes.hlines to draw its error
    limits int he vertical and horizontal direction.  As you'll see
    in the changes below, these functions now return a LineCollection
    rather than a list of lines.  The new return signature for
    errorbar is  ylins, caplines, errorcollections where
    errorcollections is a xerrcollection, yerrcollection

    Axes.vlines and Axes.hlines now create and returns a LineCollection, not a list
    of lines.  This is much faster.  The kwarg signature has changed,
    so consult the docs

    MaxNLocator accepts a new Boolean kwarg ('integer') to force
    ticks to integer locations.

    Commands that pass an argument to the Text constructor or to
    Text.set_text() now accept any object that can be converted
    with '%s'.  This affects xlabel(), title(), etc.

    Barh now takes a **kwargs dict instead of most of the old
    arguments. This helps ensure that bar and barh are kept in sync,
    but as a side effect you can no longer pass e.g., color as a
    positional argument.

    ft2font.get_charmap() now returns a dict that maps character codes
    to glyph indices (until now it was reversed)

    Moved data files into lib/matplotlib so that setuptools' develop
    mode works. Re-organized the mpl-data layout so that this source
    structure is maintained in the installation. (i.e., the 'fonts' and
    'images' sub-directories are maintained in site-packages.).
    Suggest removing site-packages/matplotlib/mpl-data and
    ~/.matplotlib/ttffont.cache before installing
