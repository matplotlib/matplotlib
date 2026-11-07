``on_select`` method for selector widgets
-----------------------------------------

All selector widgets (`~.widgets.SpanSelector`, `~.widgets.RectangleSelector`,
`~.widgets.EllipseSelector`, `~.widgets.LassoSelector`, `~.widgets.PolygonSelector`)
now support registering multiple callback functions via the new
`~.widgets._SelectorWidget.on_select` method. This is similar to how
`~.widgets.Slider.on_changed` allows multiple callbacks for slider widgets.

Previously, selectors only supported a single callback passed at initialization
via the *onselect* parameter. Now you can register additional callbacks that
will be called whenever a selection is made::

    span = SpanSelector(ax, onselect, 'horizontal')

    # Register additional callbacks
    def my_callback(vmin, vmax):
        print(f"Selection: {vmin} to {vmax}")

    cid = span.on_select(my_callback)

    # Later, disconnect if needed
    span.disconnect(cid)

The callback signature depends on the selector type:

- `~.widgets.SpanSelector`: ``func(min, max)``
- `~.widgets.RectangleSelector` and `~.widgets.EllipseSelector`:
  ``func(eclick, erelease)``
- `~.widgets.LassoSelector` and `~.widgets.PolygonSelector`: ``func(verts)``
