axline supports *transform* parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The *transform* keyword argument only applies to the points *xy1*,
*xy2*. The *slope* (if given) is always in data coordinates. This can
be used e.g. with ``ax.transAxes`` for drawing grid lines with a fixed
slope.