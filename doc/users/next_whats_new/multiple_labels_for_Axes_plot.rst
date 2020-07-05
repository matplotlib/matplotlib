An iterable object with labels can be passed to `Axes.plot()`
-------------------------------------------------------------

If multidimensional data is used for plotting, labels can be specified in
a vectorized way with an iterable object of size corresponding to the
data array shape (exactly 5 labels are expected when plotting 5 lines).
It works with `Axes.plot()` as well as with it's wrapper `plt.plot()`.

.. plot::

    from matplotlib import pyplot as plt

    x = [1, 2, 5]

    y = [[2, 4, 3],
        [4, 7, 1],
        [3, 9, 2]]

    plt.plot(x, y, label=['one', 'two', 'three'])
    plt.legend()