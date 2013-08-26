import numpy as np

from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.patheffects import (Normal, Stroke, withStroke,
                                    withSimplePatchShadow)


@image_comparison(baseline_images=['patheffect1'], remove_text=True)
def test_patheffect1():
    ax1 = plt.subplot(111)
    ax1.imshow([[1, 2], [2, 3]])
    txt = ax1.annotate("test", (1., 1.), (0., 0),
                       arrowprops=dict(arrowstyle="->",
                                       connectionstyle="angle3", lw=2),
                       size=20, ha="center",
                       path_effects=[withStroke(linewidth=3,
                                     foreground="w")])
    txt.arrow_patch.set_path_effects([Stroke(linewidth=5,
                                             foreground="w"),
                                      Normal()])

    ax1.grid(True, linestyle="-")

    pe = [withStroke(linewidth=3, foreground="w")]
    for l in ax1.get_xgridlines() + ax1.get_ygridlines():
        l.set_path_effects(pe)


@image_comparison(baseline_images=['patheffect2'], remove_text=True)
def test_patheffect2():

    ax2 = plt.subplot(111)
    arr = np.arange(25).reshape((5, 5))
    ax2.imshow(arr)
    cntr = ax2.contour(arr, colors="k")

    plt.setp(cntr.collections,
             path_effects=[withStroke(linewidth=3, foreground="w")])

    clbls = ax2.clabel(cntr, fmt="%2.0f", use_clabeltext=True)
    plt.setp(clbls,
             path_effects=[withStroke(linewidth=3, foreground="w")])


@image_comparison(baseline_images=['patheffect3'])
def test_patheffect3():

    ax3 = plt.subplot(111)
    p1, = ax3.plot([0, 1], [0, 1])
    ax3.set_title(r'testing$^{123}$',
        path_effects=[withStroke(linewidth=1, foreground="r")])
    leg = ax3.legend([p1], [r'Line 1$^2$'], fancybox=True, loc=2)
    leg.legendPatch.set_path_effects([withSimplePatchShadow()])


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
