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
    :alt: Example of default behavior (left) and axlim_clip=True (right)

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    np.random.seed(1)
    xyz = np.random.rand(25, 3)

    # Note that when a line has one vertex outside the view limits, the entire
    # line is hidden. The same is true for 3D patches (not shown).
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], '-o')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], '--*', axlim_clip=True)
    ax.set(xlim=(0.25, 0.75), ylim=(0, 1), zlim=(0, 1))
    ax.legend(['axlim_clip=False (default)', 'axlim_clip=True'])
