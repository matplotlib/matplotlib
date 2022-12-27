.. _whats-new-0-99:

What's new in Matplotlib 0.99 (Aug 29, 2009)
============================================

.. contents:: Table of Contents
   :depth: 2



New documentation
-----------------

Jae-Joon Lee has written two new guides :doc:`/tutorials/intermediate/legend_guide`
and :ref:`plotting-guide-annotation`.  Michael Sarahan has written
:doc:`/tutorials/introductory/images`.  John Hunter has written two new tutorials on
working with paths and transformations: :doc:`/tutorials/advanced/path_tutorial` and
:doc:`/tutorials/advanced/transforms_tutorial`.

.. _whats-new-mplot3d:

mplot3d
--------

Reinier Heeres has ported John Porter's mplot3d over to the new
matplotlib transformations framework, and it is now available as a
toolkit mpl_toolkits.mplot3d (which now comes standard with all mpl
installs).  See :ref:`mplot3d-examples-index` and
:doc:`/tutorials/toolkits/mplot3d`.

.. plot::

    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)

    plt.show()

.. _whats-new-axes-grid:

axes grid toolkit
-----------------

Jae-Joon Lee has added a new toolkit to ease displaying multiple images in
matplotlib, as well as some support for curvilinear grids to support
the world coordinate system. The toolkit is included standard with all
new mpl installs.   See :ref:`axes_grid1-examples-index`,
:ref:`axisartist-examples-index`, :ref:`axes_grid1_users-guide-index` and
:ref:`axisartist_users-guide-index`

.. plot::

    from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes


    def get_demo_image():
        # prepare image
        delta = 0.5

        extent = (-3, 4, -4, 3)
        x = np.arange(-3.0, 4.001, delta)
        y = np.arange(-4.0, 3.001, delta)
        X, Y = np.meshgrid(x, y)
        Z1 = np.exp(-X**2 - Y**2)
        Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
        Z = (Z1 - Z2) * 2

        return Z, extent


    def get_rgb():
        Z, extent = get_demo_image()

        Z[Z < 0] = 0.
        Z = Z / Z.max()

        R = Z[:13, :13]
        G = Z[2:, 2:]
        B = Z[:13, 2:]

        return R, G, B


    fig = plt.figure()
    ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])

    r, g, b = get_rgb()
    ax.imshow_rgb(r, g, b, origin="lower")

    ax.RGB.set_xlim(0., 9.5)
    ax.RGB.set_ylim(0.9, 10.6)

    plt.show()

.. _whats-new-spine:

Axis spine placement
--------------------

Andrew Straw has added the ability to place "axis spines" -- the lines
that denote the data limits -- in various arbitrary locations.  No
longer are your axis lines constrained to be a simple rectangle around
the figure -- you can turn on or off left, bottom, right and top, as
well as "detach" the spine to offset it away from the data.  See
:doc:`/gallery/spines/spine_placement_demo` and
:class:`matplotlib.spines.Spine`.

.. plot::

    def adjust_spines(ax, spines):
        for loc, spine in ax.spines.items():
            if loc in spines:
                spine.set_position(('outward', 10))  # outward by 10 points
            else:
                spine.set_color('none')  # don't draw spine

        # turn off ticks where there is no spine
        if 'left' in spines:
            ax.yaxis.set_ticks_position('left')
        else:
            # no yaxis ticks
            ax.yaxis.set_ticks([])

        if 'bottom' in spines:
            ax.xaxis.set_ticks_position('bottom')
        else:
            # no xaxis ticks
            ax.xaxis.set_ticks([])

    fig = plt.figure()

    x = np.linspace(0, 2*np.pi, 100)
    y = 2*np.sin(x)

    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x, y)
    adjust_spines(ax, ['left'])

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x, y)
    adjust_spines(ax, [])

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x, y)
    adjust_spines(ax, ['left', 'bottom'])

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(x, y)
    adjust_spines(ax, ['bottom'])

    plt.show()
