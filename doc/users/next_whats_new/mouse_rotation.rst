Rotating 3d plots with the mouse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rotating three-dimensional plots with the mouse has been made more intuitive.
The plot now reacts the same way to mouse movement, independent of the
particular orientation at hand; and it is possible to control all 3 rotational
degrees of freedom (azimuth, elevation, and roll). By default,
it uses a variation on Ken Shoemake's ARCBALL [1]_.
The particular style of mouse rotation can be set via
:rc:`axes3d.mouserotationstyle`.
See also :ref:`toolkit_mouse-rotation`.

To revert to the original mouse rotation style,
create a file ``matplotlibrc`` with contents::

    axes3d.mouserotationstyle: azel

To try out one of the various mouse rotation styles:

.. code::

    import matplotlib as mpl
    mpl.rcParams['axes3d.mouserotationstyle'] = 'trackball'  # 'azel', 'trackball', 'sphere', or 'arcball'

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm

    ax = plt.figure().add_subplot(projection='3d')

    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    plt.show()


.. [1] Ken Shoemake, "ARCBALL: A user interface for specifying
  three-dimensional rotation using a mouse", in Proceedings of Graphics
  Interface '92, 1992, pp. 151-156, https://doi.org/10.20380/GI1992.18
