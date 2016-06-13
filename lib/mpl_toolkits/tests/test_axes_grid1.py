from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison, cleanup

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from matplotlib.colors import LogNorm

import numpy as np


@image_comparison(baseline_images=['divider_append_axes'])
def test_divider_append_axes():

    # the random data
    np.random.seed(0)
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    fig, axScatter = plt.subplots()

    # the scatter plot:
    axScatter.scatter(x, y)

    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(axScatter)
    axHistbot = divider.append_axes("bottom", 1.2, pad=0.1, sharex=axScatter)
    axHistright = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)
    axHistleft = divider.append_axes("left", 1.2, pad=0.1, sharey=axScatter)
    axHisttop = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHisttop.hist(x, bins=bins)
    axHistbot.hist(x, bins=bins)
    axHistleft.hist(y, bins=bins, orientation='horizontal')
    axHistright.hist(y, bins=bins, orientation='horizontal')

    axHistbot.invert_yaxis()
    axHistleft.invert_xaxis()

    axHisttop.xaxis.set_ticklabels(())
    axHistbot.xaxis.set_ticklabels(())
    axHistleft.yaxis.set_ticklabels(())
    axHistright.yaxis.set_ticklabels(())


@cleanup
def test_axesgrid_colorbar_log_smoketest():
    fig = plt.figure()
    grid = AxesGrid(fig, 111,  # modified to be only subplot
                    nrows_ncols=(1, 1),
                    label_mode="L",
                    cbar_location="top",
                    cbar_mode="single",
                    )

    Z = 10000 * np.random.rand(10, 10)
    im = grid[0].imshow(Z, interpolation="nearest", norm=LogNorm())

    grid.cbar_axes[0].colorbar(im)


@image_comparison(
    baseline_images=['inset_locator'], style='default', extensions=['png'],
    remove_text=True)
def test_inset_locator():
    def get_demo_image():
        from matplotlib.cbook import get_sample_data
        import numpy as np
        f = get_sample_data("axes_grid/bivariate_normal.npy", asfileobj=False)
        z = np.load(f)
        # z is a numpy array of 15x15
        return z, (-3, 4, -4, 3)

    fig, ax = plt.subplots(figsize=[5, 4])

    # prepare the demo image
    Z, extent = get_demo_image()
    Z2 = np.zeros([150, 150], dtype="d")
    ny, nx = Z.shape
    Z2[30:30 + ny, 30:30 + nx] = Z

    # extent = [-3, 4, -4, 3]
    ax.imshow(Z2, extent=extent, interpolation="nearest",
              origin="lower")

    axins = zoomed_inset_axes(ax, 6, loc=1)  # zoom = 6
    axins.imshow(Z2, extent=extent, interpolation="nearest",
                 origin="lower")
    axins.yaxis.get_major_locator().set_params(nbins=7)
    axins.xaxis.get_major_locator().set_params(nbins=7)
    # sub region of the original image
    x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    asb = AnchoredSizeBar(ax.transData,
                          0.5,
                          '0.5',
                          loc=8,
                          pad=0.1, borderpad=0.5, sep=5,
                          frameon=False)
    ax.add_artist(asb)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
