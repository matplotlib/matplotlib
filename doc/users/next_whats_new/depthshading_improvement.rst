3D depth-shading fix
--------------------

Previously, a slightly buggy method of estimating the visual "depth" of 3D
items could lead to sudden and unexpected changes in transparency as the plot
orientation changed.

Now, the behavior has been made smooth and predictable. A new parameter
``depthshade_minalpha`` has also been added to allow users to set the minimum
transparency level. Depth-shading is an option for Patch3DCollections and
Path3DCollections, including 3D scatter plots.

The default values for ``depthshade`` and ``depthshade_minalpha`` are now also
controlled via rcParams, with values of ``True`` and ``0.3`` respectively.

A simple example:

.. plot::
    :include-source: true
    :alt: A 3D scatter plot with depth-shading enabled.

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    X = [i for i in range(10)]
    Y = [i for i in range(10)]
    Z = [i for i in range(10)]
    S = [(i + 1) * 400 for i in range(10)]

    ax.scatter(
        xs=X, ys=Y, zs=Z, s=S,
        depthshade=True,
        depthshade_minalpha=0.3,
    )
    ax.view_init(elev=10, azim=-150, roll=0)

    plt.show()
