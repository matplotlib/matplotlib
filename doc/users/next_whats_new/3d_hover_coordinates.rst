3D hover coordinates
--------------------

The x, y, z coordinates displayed in 3D plots were previously showing
nonsensical values. This has been fixed to report the coordinate on the view
pane directly beneath the mouse cursor. This is likely to be most useful when
viewing 3D plots along a primary axis direction when using an orthographic
projection, or when a 2D plot has been projected onto one of the 3D axis panes.
Note that there is still no way to directly display the coordinates of plotted
data points.
