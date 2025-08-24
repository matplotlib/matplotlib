Scroll-to-zoom in GUIs
~~~~~~~~~~~~~~~~~~~~~~

When a plot manipulation tool (pan or zoom tool) in plot windows is enabled,
a mouse scroll operation results in a zoom focussing on the mouse pointer, keeping the
aspect ratio of the axes.

There is no effect if no manipulation tool is selected. This is intentional to
keep a state in which accidental manipulation of the plot is avoided.

Zooming is currently only supported on rectilinear Axes.
