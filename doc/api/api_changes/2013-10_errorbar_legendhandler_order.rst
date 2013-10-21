Change artist associated with error bar in legend
`````````````````````````````````````````````````

Changed the order of the artist list returned by
`legend_handler.HandlerErrorbar.create_artists` so that the artist
associated with the legend marker in `legend.legendHandles` is the
`Line2D` object for the markers not the `LineCollection` object for
the error bars.
