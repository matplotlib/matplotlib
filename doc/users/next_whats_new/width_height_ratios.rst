``subplots``, ``subplot_mosaic`` accept *height_ratios* and *width_ratios* arguments
------------------------------------------------------------------------------------

The relative width and height of columns and rows in `~.Figure.subplots` and
`~.Figure.subplot_mosaic` can be controlled by passing *height_ratios* and
*width_ratios* keyword arguments to the methods.  Previously, this required
passing the ratios in *gridspec_kws* arguments.
