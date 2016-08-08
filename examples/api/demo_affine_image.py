"""
For the backends that supports draw_image with optional affine
transform (e.g., agg, ps backend), the image of the output should
have its boundary matches the red dashed rectangle. The extent of
the image without affine transform is additionally displayed with
a black solid rectangle.
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


def get_image():
    delta = 0.25
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = Z2 - Z1  # difference of Gaussians
    return Z


def plot_extent(im, rect_lw=1.5, ls="-", color="Black", transform=None):
    """Draws a rectangle denoting the extent of an image `im` altered by a
    transform `transform`. Additional segment markers going through then
    origin are also plotted.

    `rect_lw` is the linewidth parameter used to the rectangle.
    """
    x1, x2, y1, y2 = im.get_extent()
    ax = im.axes
    if transform is None:  # then no specific transform will be applied
        transform = ax.transData
    # Plot the extent rectangle
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], ls=ls, lw=rect_lw,
            color=color, transform=transform)
    # Plot the segments parallel to the rectangle sides & going through (0, 0)
    ax.plot([x1, x2], [0, 0], ls=ls, color=color, transform=transform)
    ax.plot([0, 0], [y1, y2], ls=ls, color=color, transform=transform)


if 1:

    fig, ax1 = plt.subplots(1, 1)
    Z = get_image()
    im1 = ax1.imshow(Z, interpolation='none',
                     origin='lower',
                     extent=[-2, 4, -3, 2], clip_on=True)

    # Image rotation
    trans_data2 = mtransforms.Affine2D().rotate_deg(30) + ax1.transData
    im1.set_transform(trans_data2)

    # Plot the extent of the image:
    # 1/ With the affine transform.
    plot_extent(im1, ls="--", rect_lw=3, color="Red", transform=trans_data2)
    # 2/ Without the affine transform (see `plot_extent` defaults).
    plot_extent(im1)

    ax1.set_xlim(-3, 5)
    ax1.set_ylim(-4, 4)

    plt.show()
