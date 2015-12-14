from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from mpl_toolkits.axes_grid1 import make_axes_locatable, host_subplot
from itertools import product
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


@image_comparison(baseline_images=['twin_axes_empty_and_removed'],
                  extensions=["png"], tol=1)
def test_twin_axes_empty_and_removed():
    # Purely cosmetic font changes (avoid overlap)
    matplotlib.rcParams.update({"font.size": 8})
    matplotlib.rcParams.update({"xtick.labelsize": 8})
    matplotlib.rcParams.update({"ytick.labelsize": 8})
    generators = [ "twinx", "twiny", "twin" ]
    modifiers = [ "", "host invisible", "twin removed", "twin invisible",
        "twin removed\nhost invisible" ]
    # Unmodified host subplot at the beginning for reference
    h = host_subplot(len(modifiers)+1, len(generators), 2)
    h.text(0.5, 0.5, "host_subplot", horizontalalignment="center",
        verticalalignment="center")
    # Host subplots with various modifications (twin*, visibility) applied
    for i, (mod, gen) in enumerate(product(modifiers, generators),
        len(generators)+1):
        h = host_subplot(len(modifiers)+1, len(generators), i)
        t = getattr(h, gen)()
        if "twin invisible" in mod:
            t.axis[:].set_visible(False)
        if "twin removed" in mod:
            t.remove()
        if "host invisible" in mod:
            h.axis[:].set_visible(False)
        h.text(0.5, 0.5, gen + ("\n" + mod if mod else ""),
            horizontalalignment="center", verticalalignment="center")
    plt.subplots_adjust(wspace=0.5, hspace=1)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
