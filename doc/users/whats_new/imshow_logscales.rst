Non-linears scales on image plots
---------------------------------

:func:`imshow` now draws data at the requested points in data space after the
application of non-linear scales.

The image on the left demonstrates the new, correct behavior.
The old behavior can be recreated using :func:`pcolormesh` as
demonstrated on the right.

Example
```````
::

    import numpy as np
    import matplotlib.pyplot as plt

    data = np.arange(30).reshape(5, 6)
    x = np.linspace(0, 6, 7)
    y = 10**np.linspace(0, 5, 6)
    X, Y = np.meshgrid(x, y)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

    ax1.imshow(data, aspect="auto", extent=(0, 6, 1e0, 1e5), interpolation='nearest')
    ax1.set_yscale('log')
    ax1.set_title('Using ax.imshow')

    ax2.pcolormesh(x, y, np.flipud(data))
    ax2.set_yscale('log')
    ax2.set_title('Using ax.pcolormesh')
    ax2.autoscale('tight')

    plt.show()


This can be understood by analogy to plotting a histogram with linearly spaced bins
with a logarithmic x-axis.  Equal sized bins at will be displayed as wider for small
*x* and narrower for large *x*.
