Colorbars for bivariate colormaps
---------------------------------

The new method `.Figure.colorbar_bivar` (and `.pyplot.colorbar_bivar`) adds a
colorbar for an image that uses a bivariate colormap. The colormap is shown as a
2D image, with the first data variable along the y-axis and the second along the
x-axis.

.. plot::
    :include-source: true
    :alt: An image colored with a bivariate colormap and a square colorbar on its right.

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, x)

    fig, ax = plt.subplots()
    im = ax.imshow((np.sin(X), np.cos(Y)), cmap='BiOrangeBlue')
    cbar = fig.colorbar_bivar(im, fraction=0.3)
    cbar.set_ylabel('sin(x)')
    cbar.set_xlabel('cos(y)')
