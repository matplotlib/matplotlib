Rotating selectors and aspect ratio correction
----------------------------------------------

The `~matplotlib.widgets.RectangleSelector` and
`~matplotlib.widgets.EllipseSelector` can now be rotated interactively.

The rotation is enabled when *rotate* is added to
:py:attr:`~matplotlib.widgets._SelectorWidget.state`, which can be done using
:py:meth:`~matplotlib.widgets._SelectorWidget.add_state` or by striking
the *state_modifier_keys* for *rotate* (default *r*).

The aspect ratio of the axes can now be taken into account when using the
"square" state. When *data_coordinates* is added to
:py:attr:`~matplotlib.widgets._SelectorWidget.state`, which can be done using
:py:meth:`~matplotlib.widgets._SelectorWidget.add_state` or by striking
the *state_modifier_keys* for *data_coordinates* (default *d*).
