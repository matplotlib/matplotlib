import itertools
import matplotlib.pyplot as plt
from AngleMarker import AngleMarker


def testing(size=0.25, units="axes fraction", dpi=100, fs=(6.4, 5),
            show=False):

    fig, axes = plt.subplots(2, 2, sharex="col", sharey="row", dpi=dpi,
                             figsize=fs,
                             gridspec_kw=dict(width_ratios=[1, 3],
                                              height_ratios=[3, 1]))

    def plot_angle(ax, pos, vec1, vec2, acol="C0", **kwargs):
        ax.plot([vec1[0], pos[0], vec2[0]], [vec1[1], pos[1], vec2[1]],
                color=acol)
        am = AngleMarker(pos, vec1, vec2, ax=ax, text=r"$\theta$", **kwargs)

    tx = "figsize={}, dpi={}, arcsize={} {}".format(fs, dpi, size, units)
    axes[0, 1].set_title(tx, loc="right", size=9)
    kw = dict(size=size, units=units)
    p = (.5, .2), (2, 0), (1, 1)
    plot_angle(axes[0, 0], *p, **kw)
    plot_angle(axes[0, 1], *p, **kw)
    plot_angle(axes[1, 1], *p, **kw)
    kw.update(acol="limegreen")
    plot_angle(axes[0, 0], (1.2, 0), (1, -1), (1.3, -.8), **kw)
    plot_angle(axes[1, 1], (0.2, 1), (0, 0), (.3, .2), **kw)
    plot_angle(axes[0, 1], (0.2, 0), (0, -1), (.3, -.8), **kw)
    kw.update(acol="crimson")
    plot_angle(axes[1, 0], (1, .5), (1, 1), (2, .5), **kw)

    fig.tight_layout()
    fig.savefig(tx.replace("=", "_") + ".png")
    fig.savefig(tx.replace("=", "_") + ".pdf")
    if show:
        plt.show()


s = [(0.25, "axes min"), (0.25, "axes max"),
     (0.25, "axes width"), (0.25, "axes height"),
     (100, "pixels"), (72, "points")]
d = [72, 144]
f = [(6.4, 5), (12.8, 10)]

for (size, unit), dpi, fs in itertools.product(s, d, f):
    testing(size=size, units=unit, dpi=dpi, fs=fs)
