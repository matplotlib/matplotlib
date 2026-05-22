Geometric clipping for 3D surface plots
---------------------------------------

`.Axes3D.plot_surface` now supports ``axlim_clip_mode="clip"``
when ``axlim_clip=True``. This clips surface polygons geometrically
to the axes view-limit box instead of hiding a whole polygon whenever
one of its vertices lies outside the limits.
