Add ``AnchoredDirectionArrows`` feature to mpl_toolkits
--------------------------------------------------------

A new mpl_toolkits class
:class:`~mpl_toolkits.axes_grid1.anchored_artists.AnchoredDirectionArrows`
draws a pair of orthogonal arrows to inidcate directions on a 2D plot. A
minimal working example takes in the transformation object for the coordinate
system (typically ax.transAxes), and arrow labels. There are several optional
parameters that can be used to alter layout. For example, the arrow pairs can
be rotated and the color can be changed. By default the labels and arrows have
the same color, but the class may also pass arguments for costumizing arrow
and text layout, these are passed to :class:`matplotlib.text.TextPath` and
`matplotlib.patches.FancyArrowPatch`. Location, length and width for both
arrow tail and head can be adjusted, the the direction arrows and labels can
have a frame. Padding and separation parameters can be adjusted.
