Deprecations
````````````

The following keyword arguments are deprecated:

- Passing ``shade=None`` to
  `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface` is deprecated. This was
  an unintended implementation detail with the same semantics as
  ``shade=False``. Please use the latter code instead.