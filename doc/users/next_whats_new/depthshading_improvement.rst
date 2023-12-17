Depth-shading fix and more depth-shading options
--------------------------------------------------------------

New options have been added which allow users to modify the behavior of 
depth-shading while addressing a visual bug.

Previously, a slightly buggy method of estimating the "depth" of plotted
items could lead to sudden and unexpected changes in transparency as the
plot orientation changed.

Now, the behavior has been made smooth and predictable, and the user is 
provided with three new options: whether to invert the shading, setting the
lowest acceptable alpha value (highest transparency), and whether to use
the old algorithm.

The default behavior visually matches the old algorithm: items that appear to be
"deeper" into the screen will become increasingly transparent (up to the now
user-defined limit). If the inversion option is used then items will start
at maximum transparency and become gradually opaque with increasing depth.

Note 1: depth-shading applies to Patch3DCollections and Path3DCollections,
including scatter plots.

Note 2: "depthshade=True" must still be used to enable depth-shading

A simple example:

.. plot::
    :include-source: true
    :alt: A simple example showing different behavior of depthshading, which can be modified using the provided kwargs.

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
        depthshade_minalpha=0.1,
        depthshade_inverted=True,
        depthshade_legacy=True,
    )

    plt.show()

