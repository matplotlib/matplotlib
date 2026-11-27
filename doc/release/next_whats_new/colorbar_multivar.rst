Colorbars for multivariate colormaps
------------------------------------

The new method `.Figure.colorbar_multivar` (and `.pyplot.colorbar_multivar`)
adds one colorbar per variable for an image that uses a multivariate colormap.
The colorbars can be arranged in several rows or columns with *n_major*.

This API is provisional and may change without a deprecation period.

.. plot::
    :include-source: true
    :alt: An image colored with a three-variable colormap and three colorbars on its right.

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, x)

    fig, ax = plt.subplots()
    im = ax.imshow((np.sin(X), np.cos(Y), np.sin(X + Y)), cmap='3VarAddA')
    cbars = fig.colorbar_multivar(im)
    for cbar, label in zip(cbars, ['sin(x)', 'cos(y)', 'sin(x + y)']):
        cbar.set_label(label)
