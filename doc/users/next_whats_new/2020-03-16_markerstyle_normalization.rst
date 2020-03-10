Allow for custom marker scaling
-------------------------------
`~.markers.MarkerStyle` gained a keyword argument *normalization*, which may be
set to *"none"* to allow for custom paths to not be scaled.::

    MarkerStyle(Path(...), normalization="none")

`~.markers.MarkerStyle` also gained a `~.markers.MarkerStyle.set_transform`
method to set affine transformations to existing markers.::

    m = MarkerStyle("d")
    m.set_transform(m.get_transform() + Affine2D().rotate_deg(30))
