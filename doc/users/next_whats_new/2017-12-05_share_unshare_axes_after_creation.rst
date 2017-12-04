Share and unshare `axes` after creation
---------------------------------------

`~.Axes` have `~.Axes.unshare_x_axes`, `~.Axes.unshare_y_axes`, `~.Axes.unshare_z_axes` and `~.Axes.unshare_axes` methods to unshare axes.
Similiar there are `~.Axes.share_x_axes`, `~.Axes.share_y_axes`, `~.Axes.share_z_axes` and `~.Axes.share_axes` methods to share axes.

Unshare an axis will decouple the viewlimits for further changes.
Share an axis will couple the viewlimits.