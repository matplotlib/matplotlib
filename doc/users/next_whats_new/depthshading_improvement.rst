3D depth-shading fix
--------------------

Previously, a slightly buggy method of estimating the "depth" of plotted
items could lead to sudden and unexpected changes in transparency as the
plot orientation changed.

Now, the behavior has been made smooth and predictable. A new parameter
``depthshade_minalpha`` has also been added to allow users to set the minimum
transparency level.

Depth-shading is an option for Patch3DCollections and Path3DCollections,
including 3D scatter plots. Depth-shading is still off by default, and
``depthshade=True`` must still be used to enable it.

A simple example:

.. plot::
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    X = [i for i in range(10)]
    Y = [i for i in range(10)]
    Z = [i for i in range(10)]
    S = [(i + 1) * 400 for i in range(10)]

    ax.scatter(
        xs=X,
        ys=Y,
        zs=Z,
        s=S,
        depthshade=True,
        depthshade_minalpha=0.3,
    )

    plt.show()
