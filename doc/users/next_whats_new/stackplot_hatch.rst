``hatch`` parameter for stackplot
-------------------------------------------

The `~.Axes.stackplot` *hatch* parameter now accepts a list of strings describing hatching styles that will be applied sequentially to the layers in the stack:

.. plot::
    :include-source: true
    :alt: Two charts, identified as ax1 and ax2, showing "stackplots", i.e. one-dimensional distributions of data stacked on top of one another. The first plot, ax1 has cross-hatching on all slices, having been given a single string as the "hatch" argument. The second plot, ax2 has different styles of hatching on each slice - diagonal hatching in opposite directions on the first two slices, cross-hatching on the third slice, and open circles on the fourth.

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))

    cols = 10
    rows = 4
    data = (
    np.reshape(np.arange(0, cols, 1), (1, -1)) ** 2
    + np.reshape(np.arange(0, rows), (-1, 1))
    + np.random.random((rows, cols))*5
    )
    x = range(data.shape[1])
    ax1.stackplot(x, data, hatch="x")
    ax2.stackplot(x, data, hatch=["//","\\","x","o"])

    ax1.set_title("hatch='x'")
    ax2.set_title("hatch=['//','\\\\','x','o']")

    plt.show()
