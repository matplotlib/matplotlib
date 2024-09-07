Specifying a single color in ``contour`` and ``contourf``
---------------------------------------------------------

`~.Axes.contour` and `~.Axes.contourf` previously accepted a single color
provided it was expressed as a string.  This restriction has now been removed
and a single color in any format described in the :ref:`colors_def` tutorial
may be passed.

.. plot::
    :include-source: true
    :alt: Two-panel example contour plots.  The left panel has all transparent red contours.  The right panel has all dark blue contours.

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 3))
    z = [[0, 1], [1, 2]]

    ax1.contour(z, colors=('r', 0.4))
    ax2.contour(z, colors=(0.1, 0.2, 0.5))

    plt.show()
