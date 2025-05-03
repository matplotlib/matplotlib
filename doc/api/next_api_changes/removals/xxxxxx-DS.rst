``apply_theta_transforms`` option in ``PolarTransform``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applying theta transforms in `~matplotlib.projections.polar.PolarTransform` and
`~matplotlib.projections.polar.InvertedPolarTransform` has been removed, and
the ``apply_theta_transforms`` keyword argument removed from both classes.

If you need to retain the behaviour where theta values
are transformed, chain the ``PolarTransform`` with a `~matplotlib.transforms.Affine2D`
transform that performs the theta shift and/or sign shift.
