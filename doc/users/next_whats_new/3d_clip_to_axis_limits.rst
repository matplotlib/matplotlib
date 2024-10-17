Data in 3D plots can now be dynamically clipped to the axes view limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All 3D plotting functions now support the *axlim_clip* keyword argument, which
will clip the data to the axes view limits, hiding all data outside those
bounds. This clipping will be dynamically applied in real time while panning
and zooming.

Please note that if one vertex of a line segment or 3D patch is clipped, then
the entire segment or patch will be hidden. Not being able to show partial
lines or patches such that they are "smoothly" cut off at the boundaries of the
view box is a limitation of the current renderer.

.. plot::
    :include-source: true
    :alt: Example of default behavior (blue) and axlim_clip=True (orange)

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x = np.arange(-5, 5, 0.5)
    y = np.arange(-5, 5, 0.5)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    # Note that when a line has one vertex outside the view limits, the entire
    # line is hidden. The same is true for 3D patches (not shown).
    # In this example, data where x < 0 or z > 0.5 is clipped.
    ax.plot_wireframe(X, Y, Z, color='C0')
    ax.plot_wireframe(X, Y, Z, color='C1', axlim_clip=True)
    ax.set(xlim=(0, 10), ylim=(-5, 5), zlim=(-1, 0.5))
    ax.legend(['axlim_clip=False (default)', 'axlim_clip=True'])
