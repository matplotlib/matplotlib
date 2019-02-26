Changes for 0.85
================

.. code-block:: text

    Made xtick and ytick separate props in rc

    made pos=None the default for tick formatters rather than 0 to
    indicate "not supplied"

    Removed "feature" of minor ticks which prevents them from
    overlapping major ticks.  Often you want major and minor ticks at
    the same place, and can offset the major ticks with the pad.  This
    could be made configurable

    Changed the internal structure of contour.py to a more OO style.
    Calls to contour or contourf in axes.py or pylab.py now return
    a ContourSet object which contains references to the
    LineCollections or PolyCollections created by the call,
    as well as the configuration variables that were used.
    The ContourSet object is a "mappable" if a colormap was used.

    Added a clip_ends kwarg to contourf. From the docstring:
             * clip_ends = True
               If False, the limits for color scaling are set to the
               minimum and maximum contour levels.
               True (default) clips the scaling limits.  Example:
               if the contour boundaries are V = [-100, 2, 1, 0, 1, 2, 100],
               then the scaling limits will be [-100, 100] if clip_ends
               is False, and [-3, 3] if clip_ends is True.
    Added kwargs linewidths, antialiased, and nchunk to contourf.  These
    are experimental; see the docstring.

    Changed Figure.colorbar():
        kw argument order changed;
        if mappable arg is a non-filled ContourSet, colorbar() shows
                lines instead hof polygons.
        if mappable arg is a filled ContourSet with clip_ends=True,
                the endpoints are not labelled, so as to give the
                correct impression of open-endedness.

    Changed LineCollection.get_linewidths to get_linewidth, for
    consistency.
