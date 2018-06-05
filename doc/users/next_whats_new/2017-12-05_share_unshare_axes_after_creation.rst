Share and unshare `axes` after creation
---------------------------------------

`matplotlib.axes.Axes` have `matplotlib.axes.Axes.unshare_x_axes`,
`matplotlib.axes.Axes.unshare_y_axes`, `matplotlib.axes.Axes.unshare_z_axes`
and `matplotlib.axes.Axes.unshare_axes` methods to unshare axes.
Similiar there are `matplotlib.axes.Axes.share_x_axes`,
`matplotlib.axes.Axes.share_y_axes`, `matplotlib.axes.Axes.share_z_axes` and
`matplotlib.axes.Axes.share_axes` methods to share axes.

Unshare an axis will decouple the viewlimits for further changes.
Share an axis will couple the viewlimits.