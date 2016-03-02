from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib


def draw_arrow(ax, t, r):
    ax.annotate('', xy=(0.5, 0.5 + r), xytext=(0.5, 0.5), size=30,
                arrowprops=dict(arrowstyle=t,
                                fc="b", ec='k'))


@image_comparison(baseline_images=['fancyarrow_test_image'])
def test_fancyarrow():
    # Added 0 to test division by zero error described in issue 3930
    r = [0.4, 0.3, 0.2, 0.1, 0]
    t = ["fancy", "simple", matplotlib.patches.ArrowStyle.Fancy()]

    fig, axes = plt.subplots(len(t), len(r), squeeze=False,
                             subplot_kw=dict(aspect=True),
                             figsize=(8, 4.5))

    for i_r, r1 in enumerate(r):
        for i_t, t1 in enumerate(t):
            ax = axes[i_t, i_r]
            draw_arrow(ax, t1, r1)
            ax.tick_params(labelleft=False, labelbottom=False)


@image_comparison(baseline_images=['boxarrow_test_image'], extensions=['png'])
def test_boxarrow():

    styles = matplotlib.patches.BoxStyle.get_styles()

    n = len(styles)
    spacing = 1.2

    figheight = (n * spacing + .5)
    fig1 = plt.figure(1, figsize=(4 / 1.5, figheight / 1.5))

    fontsize = 0.3 * 72

    for i, stylename in enumerate(sorted(styles.keys())):
        fig1.text(0.5, ((n - i) * spacing - 0.5)/figheight, stylename,
                  ha="center",
                  size=fontsize,
                  transform=fig1.transFigure,
                  bbox=dict(boxstyle=stylename, fc="w", ec="k"))


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
