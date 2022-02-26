Allow setting figure label size and weight globally and separately from title
-----------------------------------------------------------------------------

The figure labels, ``Figure.supxlabel`` and ``Figure.supylabel``, size and
weight can be set separately from the figure title. Use :rc:`figure.labelsize`
and :rc:`figure.labelweight`.

Note that if you have locally changed :rc:`figure.titlesize` or
:rc:`figure.titleweight`, you must now also change the introduced parameters
for a consistent result.
