from matplotlib import patheffects


__mpl_style__ = {
    "axes.edgecolor": "black",
    "axes.grid": False,
    "axes.linewidth": 1.5,
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "font.family": ["xkcd", "Humor Sans", "Comic Sans MS"],
    "font.size": 14.0,
    "grid.linewidth": 0.0,
    "lines.linewidth": 2.0,
    "path.effects": [patheffects.withStroke(linewidth=4, foreground="w")],
    "path.sketch": (1, 100, 2),
    "text.usetex": False,
    "xtick.major.size": 8,
    "xtick.major.width": 3,
    "ytick.major.size": 8,
    "ytick.major.width": 3,
}
